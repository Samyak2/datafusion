// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Benchmark-only runtime choice of a hash join's distribution.

use std::fmt;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use arrow::record_batch::RecordBatch;
use datafusion::common::runtime::SpawnedTask;
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{DataFusionError, Result, internal_err};
use datafusion::execution::TaskContext;
use datafusion::execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_optimizer::ensure_requirements::EnsureRequirements;
use datafusion::physical_optimizer::sanity_checker::SanityCheckPlan;
use datafusion::physical_plan::coop::make_cooperative;
use datafusion::physical_plan::execution_plan::{
    EmissionType, EvaluationType, SchedulingType,
};
use datafusion::physical_plan::joins::{HashJoinExec, PartitionMode};
use datafusion::physical_plan::metrics::MetricsSet;
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, Distribution, ExecutionPlan, ExecutionPlanProperties,
    InputDistributionRequirements, Partitioning, PlanProperties,
    SendableRecordBatchStream, StageBoundary,
};
use futures::future::{BoxFuture, Shared};
use futures::{FutureExt, StreamExt, TryStreamExt, stream};
use tokio::sync::{Notify, watch};

/// Install a runtime distribution choice for eligible collect-left hash joins.
///
/// Input stages start during execution. After buffering the complete build, keep
/// collect-left below `threshold_bytes` and hash-partition both inputs otherwise.
/// Bytes are the total `RecordBatch::get_array_memory_size` across build partitions;
/// shared allocations can be counted repeatedly, and hash-table size is excluded.
/// Parent requirements are repaired before execution for either output distribution.
///
/// shortcut: buffer the full build in memory without spilling; keep this experimental
/// operator in benchmarks until incremental decisions and spilling are supported.
pub fn install_adaptive_joins(
    plan: Arc<dyn ExecutionPlan>,
    context: &TaskContext,
    threshold_bytes: usize,
) -> Result<Arc<dyn ExecutionPlan>> {
    let transformed = plan.transform_up(|plan| {
        let Some(join) = plan.downcast_ref::<HashJoinExec>() else {
            return Ok(Transformed::no(plan));
        };
        // Null-aware anti joins require a global build-side null check.
        if *join.partition_mode() != PartitionMode::CollectLeft || join.null_aware {
            return Ok(Transformed::no(plan));
        }
        Ok(Transformed::yes(
            Arc::new(AdaptiveJoinExec::new(plan, threshold_bytes)?)
                as Arc<dyn ExecutionPlan>,
        ))
    })?;
    if !transformed.transformed {
        return Ok(transformed.data);
    }
    let options = context.session_config().options();
    let plan = EnsureRequirements::new().optimize(transformed.data, options)?;
    SanityCheckPlan::new().optimize(plan, options)
}

type Decision = Shared<
    BoxFuture<'static, std::result::Result<Arc<dyn ExecutionPlan>, Arc<DataFusionError>>>,
>;

struct AdaptiveJoinExec {
    join: Arc<dyn ExecutionPlan>,
    properties: Arc<PlanProperties>,
    threshold_bytes: usize,
    decision: Mutex<Option<Decision>>,
}

impl fmt::Debug for AdaptiveJoinExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AdaptiveJoinExec")
            .field("join", &self.join)
            .field("threshold_bytes", &self.threshold_bytes)
            .finish_non_exhaustive()
    }
}

impl AdaptiveJoinExec {
    fn new(join: Arc<dyn ExecutionPlan>, threshold_bytes: usize) -> Result<Self> {
        let Some(hash_join) = join.downcast_ref::<HashJoinExec>() else {
            return internal_err!("Adaptive join requires a HashJoinExec");
        };
        let partitions = hash_join.right().output_partitioning().partition_count();
        // Neither branch promises common output ordering or hash partitioning.
        let properties = Arc::new(
            join.properties()
                .as_ref()
                .clone()
                .with_eq_properties(EquivalenceProperties::new(join.schema()))
                .with_partitioning(Partitioning::UnknownPartitioning(partitions)),
        );
        Ok(Self {
            join,
            properties,
            threshold_bytes,
            decision: Mutex::new(None),
        })
    }

    fn start(&self, context: &Arc<TaskContext>) -> Result<Decision> {
        let mut decision = self.decision.lock().unwrap();
        if let Some(decision) = decision.as_ref() {
            return Ok(decision.clone());
        }
        let join = self.join.downcast_ref::<HashJoinExec>().unwrap();
        let boundary = Arc::new(BuildStageBoundary::new(Arc::clone(join.left())));
        for partition in 0..boundary.properties().partitioning.partition_count() {
            boundary.prime(partition, Arc::clone(context))?;
        }
        // Calling execute here preserves Coalesce/Repartition task startup in
        // independent probe subtrees while this join's build is still running.
        let probe = Arc::new(StartedProbeExec::new(join.right(), context)?);
        let original = Arc::clone(&self.join);
        let threshold_bytes = self.threshold_bytes;
        let future = async move {
            boundary.wait_ready().await;
            let partitioned = boundary.build_bytes()? >= threshold_bytes;
            let join = original.downcast_ref::<HashJoinExec>().unwrap();
            let mut left: Arc<dyn ExecutionPlan> = boundary.clone();
            let mut right: Arc<dyn ExecutionPlan> = probe;
            let mode = if partitioned {
                // A fixed output count lets the installed parent requirements
                // remain valid in both branches without execution-time repair.
                let partitions = right.output_partitioning().partition_count();
                let (left_keys, right_keys): (Vec<_>, Vec<_>) =
                    join.on().iter().cloned().unzip();
                left = Arc::new(RepartitionExec::try_new(
                    left,
                    Partitioning::Hash(left_keys, partitions),
                )?);
                right = Arc::new(RepartitionExec::try_new(
                    right,
                    Partitioning::Hash(right_keys, partitions),
                )?);
                PartitionMode::Partitioned
            } else {
                PartitionMode::CollectLeft
            };
            // The original join has not executed. Its builder keeps the shared
            // dynamic filter connected to the already-started probe subtree.
            let selected = join
                .builder()
                .with_new_children(vec![left, right])?
                .with_partition_mode(mode)
                .recompute_properties()
                .build_exec()?;
            boundary.release();
            Ok(selected)
        }
        .map(|result: Result<_>| result.map_err(Arc::new))
        .boxed()
        .shared();
        *decision = Some(future.clone());
        Ok(future)
    }
}

impl DisplayAs for AdaptiveJoinExec {
    fn fmt_as(&self, _: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "AdaptiveHashJoin: threshold_bytes={}",
            self.threshold_bytes
        )?;
        let decision = self.decision.lock().unwrap();
        if let Some(Ok(selected)) = decision.as_ref().and_then(Decision::peek) {
            let join = selected.downcast_ref::<HashJoinExec>().unwrap();
            write!(f, ", mode={:?}", join.partition_mode())?;
        }
        Ok(())
    }
}

impl ExecutionPlan for AdaptiveJoinExec {
    fn name(&self) -> &str {
        Self::static_name()
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        self.join.children()
    }

    fn input_distribution_requirements(&self) -> InputDistributionRequirements {
        InputDistributionRequirements::new(vec![
            Distribution::SinglePartition,
            Distribution::UnspecifiedDistribution,
        ])
    }

    fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        self.join.apply_expressions(f)
    }

    fn dynamic_expressions_produced(&self) -> Vec<Arc<dyn PhysicalExpr>> {
        self.join.dynamic_expressions_produced()
    }

    fn metrics(&self) -> Option<MetricsSet> {
        self.join.metrics()
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let join = self
            .join
            .downcast_ref::<HashJoinExec>()
            .unwrap()
            .builder()
            .with_new_children(children)?
            .recompute_properties()
            .build_exec()?;
        Ok(Arc::new(Self::new(join, self.threshold_bytes)?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition >= self.properties.partitioning.partition_count() {
            return internal_err!("Invalid adaptive join partition {partition}");
        }
        let decision = self.start(&context)?;
        let output = stream::once(async move {
            let selected = decision.await.map_err(DataFusionError::Shared)?;
            selected.execute(partition, context)
        })
        .try_flatten();
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            output,
        )))
    }
}

/// Each probe partition is executed once, before the runtime choice is awaited.
struct StartedProbeExec {
    properties: Arc<PlanProperties>,
    streams: Vec<Mutex<Option<SendableRecordBatchStream>>>,
}

impl StartedProbeExec {
    fn new(input: &Arc<dyn ExecutionPlan>, context: &Arc<TaskContext>) -> Result<Self> {
        let streams = (0..input.output_partitioning().partition_count())
            .map(|partition| {
                input
                    .execute(partition, Arc::clone(context))
                    .map(|stream| Mutex::new(Some(stream)))
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            properties: Arc::clone(input.properties()),
            streams,
        })
    }
}

impl fmt::Debug for StartedProbeExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StartedProbeExec").finish_non_exhaustive()
    }
}

impl DisplayAs for StartedProbeExec {
    fn fmt_as(&self, _: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "StartedAdaptiveProbe")
    }
}

impl ExecutionPlan for StartedProbeExec {
    fn name(&self) -> &str {
        Self::static_name()
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn apply_expressions(
        &self,
        _: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return internal_err!("Started probe has no replaceable children");
        }
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        _: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let Some(stream) = self.streams.get(partition) else {
            return internal_err!("Invalid started-probe partition {partition}");
        };
        stream.lock().unwrap().take().ok_or_else(|| {
            DataFusionError::Internal(
                "Adaptive probe partition executed twice".to_string(),
            )
        })
    }
}

#[derive(Debug)]
struct BufferedBuild {
    batches: Vec<std::result::Result<RecordBatch, Arc<DataFusionError>>>,
    reservation: MemoryReservation,
}

#[derive(Debug, Default)]
struct BuildPartition {
    task: Mutex<Option<SpawnedTask<()>>>,
    buffered: Arc<Mutex<Option<BufferedBuild>>>,
    ready: Arc<AtomicBool>,
}

/// A buffered build is a leaf so optimizer passes cannot rewrite its input.
#[derive(Debug)]
struct BuildStageBoundary {
    input: Arc<dyn ExecutionPlan>,
    properties: Arc<PlanProperties>,
    partitions: Vec<BuildPartition>,
    released: watch::Sender<bool>,
    ready_notification: Arc<Notify>,
}

impl BuildStageBoundary {
    fn new(input: Arc<dyn ExecutionPlan>) -> Self {
        let partitions = (0..input.properties().partitioning.partition_count())
            .map(|_| BuildPartition::default())
            .collect();
        let properties = Arc::new(
            input
                .properties()
                .as_ref()
                .clone()
                .with_emission_type(EmissionType::Final)
                .with_evaluation_type(EvaluationType::Eager)
                .with_scheduling_type(SchedulingType::Cooperative),
        );
        Self {
            input,
            properties,
            partitions,
            released: watch::channel(false).0,
            ready_notification: Arc::new(Notify::new()),
        }
    }

    async fn wait_ready(&self) {
        while !(0..self.partitions.len()).all(|p| self.is_ready(p)) {
            // One waiter; notify_one retains a permit across the readiness check.
            self.ready_notification.notified().await;
        }
    }

    /// Total across every build partition. This matches in-memory batch statistics:
    /// shared buffers can be counted repeatedly; hash-table and serialized sizes differ.
    fn build_bytes(&self) -> Result<usize> {
        let mut bytes = 0;
        for partition in &self.partitions {
            let buffered = partition.buffered.lock().unwrap();
            let Some(buffered) = buffered.as_ref() else {
                return internal_err!("Build stage is not ready");
            };
            for batch in &buffered.batches {
                match batch {
                    Ok(batch) => {
                        bytes += batch.get_array_memory_size();
                    }
                    Err(error) => return Err(DataFusionError::Shared(Arc::clone(error))),
                }
            }
        }
        Ok(bytes)
    }
}

impl StageBoundary for BuildStageBoundary {
    fn prime(&self, partition: usize, context: Arc<TaskContext>) -> Result<()> {
        let Some(state) = self.partitions.get(partition) else {
            return internal_err!("Invalid build-stage partition {partition}");
        };
        let mut task = state.task.lock().unwrap();
        if task.is_some() {
            return Ok(());
        }
        let mut input =
            make_cooperative(self.input.execute(partition, Arc::clone(&context))?);
        let buffered = Arc::clone(&state.buffered);
        let ready = Arc::clone(&state.ready);
        let ready_notification = self.ready_notification.clone();
        let reservation =
            MemoryConsumer::new(format!("adaptive build stage[{partition}]"))
                .register(context.memory_pool());
        *task = Some(SpawnedTask::spawn(async move {
            let mut batches = Vec::new();
            loop {
                let item = match AssertUnwindSafe(input.next()).catch_unwind().await {
                    Ok(Some(item)) => item,
                    Ok(None) => break,
                    Err(_) => internal_err!("Adaptive build stage input panicked"),
                };
                let item = item.and_then(|batch| {
                    reservation.try_grow(batch.get_array_memory_size())?;
                    Ok(batch)
                });
                let terminal_error = item.is_err();
                batches.push(item.map_err(Arc::new));
                if terminal_error {
                    break;
                }
            }
            *buffered.lock().unwrap() = Some(BufferedBuild {
                batches,
                reservation,
            });
            ready.store(true, Ordering::Release);
            ready_notification.notify_one();
        }));
        Ok(())
    }

    fn is_ready(&self, partition: usize) -> bool {
        self.partitions
            .get(partition)
            .is_some_and(|p| p.ready.load(Ordering::Acquire))
    }

    fn release(&self) {
        self.released.send_replace(true);
    }
}

impl DisplayAs for BuildStageBoundary {
    fn fmt_as(&self, _: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "AdaptiveBuildStage")
    }
}

impl ExecutionPlan for BuildStageBoundary {
    fn name(&self) -> &str {
        Self::static_name()
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn apply_expressions(
        &self,
        _: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return internal_err!("A completed build stage has no replaceable children");
        }
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        _: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let Some(state) = self.partitions.get(partition) else {
            return internal_err!("Invalid build-stage partition {partition}");
        };
        let buffered = Arc::clone(&state.buffered);
        let mut released = self.released.subscribe();
        let stream = stream::once(async move {
            while !*released.borrow_and_update() {
                released.changed().await.map_err(|_| {
                    DataFusionError::Execution("Build stage was cancelled".to_string())
                })?;
            }
            let Some(buffered) = buffered.lock().unwrap().take() else {
                return internal_err!(
                    "Build-stage partition is not ready or executed twice"
                );
            };
            let BufferedBuild {
                batches,
                reservation,
            } = buffered;
            Ok(stream::iter(batches).map(move |batch| {
                batch.map_err(DataFusionError::Shared).inspect(|batch| {
                    reservation.shrink(batch.get_array_memory_size());
                })
            }))
        })
        .try_flatten();
        Ok(make_cooperative(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            stream,
        ))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use arrow::array::{ArrayRef, Int32Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use datafusion::common::{JoinType, NullEquality, ScalarValue};
    use datafusion::datasource::MemTable;
    use datafusion::execution::memory_pool::GreedyMemoryPool;
    use datafusion::execution::runtime_env::RuntimeEnvBuilder;
    use datafusion::physical_expr::expressions::Column;
    use datafusion::physical_plan::collect;
    use datafusion::prelude::{SessionConfig, SessionContext};
    use futures::future::try_join_all;

    fn context() -> Result<SessionContext> {
        let context = SessionContext::new_with_config(
            SessionConfig::new().with_target_partitions(2),
        );
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int32, true),
            Field::new("v", DataType::Int32, false),
        ]));
        for (name, keys) in [
            ("l", vec![Some(1), Some(2), None, Some(2), Some(4), Some(7)]),
            ("r", vec![Some(2), Some(3), None, Some(2), Some(4), Some(8)]),
        ] {
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int32Array::from(keys)) as ArrayRef,
                    Arc::new(Int32Array::from(vec![10, 20, 30, 40, 50, 60])) as ArrayRef,
                ],
            )?;
            context.register_table(
                name,
                Arc::new(MemTable::try_new(
                    Arc::clone(&schema),
                    vec![vec![batch.slice(0, 3)], vec![batch.slice(3, 3)]],
                )?),
            )?;
        }
        Ok(context)
    }

    async fn plan(context: &SessionContext, sql: &str) -> Result<Arc<dyn ExecutionPlan>> {
        let plan = context.sql(sql).await?.create_physical_plan().await?;
        let plan = plan
            .transform_up(|plan| {
                if let Some(join) = plan.downcast_ref::<HashJoinExec>() {
                    return Ok(Transformed::yes(
                        join.builder()
                            .with_partition_mode(PartitionMode::CollectLeft)
                            .recompute_properties()
                            .build_exec()?,
                    ));
                }
                Ok(Transformed::no(plan))
            })?
            .data;
        EnsureRequirements::new().optimize(plan, context.state().config_options())
    }

    fn ordered_rows(batches: &[RecordBatch]) -> Result<Vec<Vec<String>>> {
        let mut rows = Vec::new();
        for batch in batches {
            for row in 0..batch.num_rows() {
                rows.push(
                    batch
                        .columns()
                        .iter()
                        .map(|column| {
                            ScalarValue::try_from_array(column, row)
                                .map(|value| value.to_string())
                        })
                        .collect::<Result<_>>()?,
                );
            }
        }
        Ok(rows)
    }

    fn rows(batches: &[RecordBatch]) -> Result<Vec<Vec<String>>> {
        let mut rows = ordered_rows(batches)?;
        rows.sort();
        Ok(rows)
    }

    fn adaptive_nodes(
        plan: &Arc<dyn ExecutionPlan>,
    ) -> Result<Vec<Arc<dyn ExecutionPlan>>> {
        let mut nodes = Vec::new();
        plan.apply(|node| {
            if node.downcast_ref::<AdaptiveJoinExec>().is_some() {
                nodes.push(Arc::clone(node));
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
        Ok(nodes)
    }

    fn assert_mode(nodes: &[Arc<dyn ExecutionPlan>], expected: PartitionMode) {
        for node in nodes {
            let join = node.downcast_ref::<AdaptiveJoinExec>().unwrap();
            let decision = join.decision.lock().unwrap();
            let selected = decision.as_ref().unwrap().peek().unwrap().as_ref().unwrap();
            assert_eq!(
                *selected
                    .downcast_ref::<HashJoinExec>()
                    .unwrap()
                    .partition_mode(),
                expected
            );
        }
    }

    fn expected_mode(threshold: usize) -> PartitionMode {
        if threshold == 0 {
            PartitionMode::Partitioned
        } else {
            PartitionMode::CollectLeft
        }
    }

    #[tokio::test]
    async fn modes_preserve_join_semantics() -> Result<()> {
        for sql in [
            "SELECT l.v, r.k FROM l JOIN r ON l.k = r.k AND l.v + r.v > 35",
            "SELECT l.v, r.k FROM l LEFT JOIN r ON l.k = r.k AND l.v + r.v > 35",
            "SELECT l.v, r.k FROM l RIGHT JOIN r ON l.k = r.k AND l.v + r.v > 35",
            "SELECT l.v, r.k FROM l FULL JOIN r ON l.k = r.k AND l.v + r.v > 35",
            "SELECT l.k, l.v FROM l LEFT SEMI JOIN r ON l.k = r.k",
            "SELECT l.k, l.v FROM l LEFT ANTI JOIN r ON l.k = r.k",
            "SELECT r.k, r.v FROM l RIGHT SEMI JOIN r ON l.k = r.k",
            "SELECT r.k, r.v FROM l RIGHT ANTI JOIN r ON l.k = r.k",
            "SELECT l.v, r.k FROM l JOIN r ON l.k IS NOT DISTINCT FROM r.k",
        ] {
            let context = context()?;
            let expected =
                rows(&collect(plan(&context, sql).await?, context.task_ctx()).await?)?;
            for threshold in [0, usize::MAX] {
                let plan = install_adaptive_joins(
                    plan(&context, sql).await?,
                    &context.task_ctx(),
                    threshold,
                )?;
                let nodes = adaptive_nodes(&plan)?;
                assert_eq!(nodes.len(), 1, "{sql}");
                assert!(
                    nodes[0]
                        .downcast_ref::<AdaptiveJoinExec>()
                        .unwrap()
                        .decision
                        .lock()
                        .unwrap()
                        .is_none()
                );
                let actual = rows(&collect(plan, context.task_ctx()).await?)?;
                assert_eq!(actual, expected, "{sql}; threshold={threshold}");
                assert_mode(&nodes, expected_mode(threshold));
                drop(nodes);
                assert_eq!(context.task_ctx().memory_pool().reserved(), 0);
            }
        }
        Ok(())
    }

    #[tokio::test]
    async fn byte_threshold_counts_total_width_and_switches_at_equality() -> Result<()> {
        let mut narrow_bytes = 0;
        for width in [1, 1024] {
            let context = SessionContext::new_with_config(
                SessionConfig::new().with_target_partitions(2),
            );
            let build_batches = (1..=2)
                .map(|key| {
                    RecordBatch::try_from_iter(vec![
                        ("k", Arc::new(Int32Array::from(vec![key])) as ArrayRef),
                        (
                            "payload",
                            Arc::new(StringArray::from(vec!["x".repeat(width)]))
                                as ArrayRef,
                        ),
                    ])
                })
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let build_bytes = build_batches
                .iter()
                .map(RecordBatch::get_array_memory_size)
                .sum::<usize>();
            if width == 1 {
                narrow_bytes = build_bytes;
            } else {
                assert!(build_bytes > narrow_bytes + 1);
            }
            context.register_table(
                "build",
                Arc::new(MemTable::try_new(
                    build_batches[0].schema(),
                    build_batches.into_iter().map(|batch| vec![batch]).collect(),
                )?),
            )?;
            context.register_batch(
                "probe",
                RecordBatch::try_from_iter(vec![(
                    "k",
                    Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
                )])?,
            )?;
            for threshold in [narrow_bytes + 1, build_bytes] {
                // Construct the chosen build side directly, retaining the payload and
                // both partitions so the test measures total bytes, not bytes per partition.
                let join: Arc<dyn ExecutionPlan> = Arc::new(HashJoinExec::try_new(
                    context.table("build").await?.create_physical_plan().await?,
                    context.table("probe").await?.create_physical_plan().await?,
                    vec![(Arc::new(Column::new("k", 0)), Arc::new(Column::new("k", 0)))],
                    None,
                    &JoinType::Inner,
                    None,
                    PartitionMode::CollectLeft,
                    NullEquality::NullEqualsNothing,
                    false,
                )?);
                let plan = install_adaptive_joins(join, &context.task_ctx(), threshold)?;
                let nodes = adaptive_nodes(&plan)?;
                assert_eq!(nodes.len(), 1);
                let result = collect(plan, context.task_ctx()).await?;
                assert_eq!(result.iter().map(RecordBatch::num_rows).sum::<usize>(), 2);
                assert_mode(
                    &nodes,
                    if build_bytes >= threshold {
                        PartitionMode::Partitioned
                    } else {
                        PartitionMode::CollectLeft
                    },
                );
                drop(nodes);
                assert_eq!(context.task_ctx().memory_pool().reserved(), 0);
            }
        }
        Ok(())
    }

    #[tokio::test]
    async fn nested_and_scalar_joins_preserve_requirements() -> Result<()> {
        let context = context()?;
        for sql in [
            "SELECT x.k, SUM(x.v + r.v) AS total FROM \
             (SELECT l.k, SUM(l.v) AS v FROM l JOIN r ON l.k = r.k GROUP BY l.k) x \
             JOIN r ON x.k = r.k GROUP BY x.k ORDER BY total DESC",
            "SELECT l.k, r.v FROM l JOIN r ON l.k = r.k \
             WHERE l.v < (SELECT SUM(l.v) FROM l JOIN r ON l.k = r.k) \
             AND l.k <= (SELECT MAX(r.k) FROM r JOIN l ON l.k = r.k) ORDER BY l.k, r.v",
        ] {
            // Keep order here: conservative properties must repair the parent sort too.
            let baseline =
                collect(plan(&context, sql).await?, context.task_ctx()).await?;
            for threshold in [0, usize::MAX] {
                let plan = install_adaptive_joins(
                    plan(&context, sql).await?,
                    &context.task_ctx(),
                    threshold,
                )?;
                let nodes = adaptive_nodes(&plan)?;
                assert!(nodes.len() >= 2);
                let actual = collect(plan, context.task_ctx()).await?;
                assert_eq!(rows(&actual)?, rows(&baseline)?, "{sql}");
                assert_eq!(
                    ordered_rows(&actual)?,
                    ordered_rows(&baseline)?,
                    "{sql}; threshold={threshold:?}"
                );
                assert_mode(&nodes, expected_mode(threshold));
                drop(nodes);
                assert_eq!(context.task_ctx().memory_pool().reserved(), 0);
            }
        }
        Ok(())
    }

    #[tokio::test]
    async fn null_aware_join_stays_collect_left() -> Result<()> {
        let context = context()?;
        let sql = "SELECT k FROM l WHERE k NOT IN (SELECT k FROM r)";
        let expected =
            rows(&collect(plan(&context, sql).await?, context.task_ctx()).await?)?;
        let original = plan(&context, sql).await?;
        let plan = install_adaptive_joins(Arc::clone(&original), &context.task_ctx(), 0)?;
        assert!(Arc::ptr_eq(&plan, &original));
        assert!(adaptive_nodes(&plan)?.is_empty());
        assert_eq!(rows(&collect(plan, context.task_ctx()).await?)?, expected);
        Ok(())
    }

    #[derive(Debug)]
    struct StartsExec {
        input: Arc<dyn ExecutionPlan>,
        starts: Arc<AtomicUsize>,
    }

    impl DisplayAs for StartsExec {
        fn fmt_as(&self, _: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "StartsExec")
        }
    }

    impl ExecutionPlan for StartsExec {
        fn name(&self) -> &str {
            Self::static_name()
        }
        fn properties(&self) -> &Arc<PlanProperties> {
            self.input.properties()
        }
        fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
            vec![&self.input]
        }
        fn apply_expressions(
            &self,
            _: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
        ) -> Result<TreeNodeRecursion> {
            Ok(TreeNodeRecursion::Continue)
        }
        fn with_new_children(
            self: Arc<Self>,
            mut children: Vec<Arc<dyn ExecutionPlan>>,
        ) -> Result<Arc<dyn ExecutionPlan>> {
            Ok(Arc::new(Self {
                input: children.remove(0),
                starts: Arc::clone(&self.starts),
            }))
        }
        fn execute(
            &self,
            partition: usize,
            context: Arc<TaskContext>,
        ) -> Result<SendableRecordBatchStream> {
            self.starts.fetch_add(1, Ordering::SeqCst);
            self.input.execute(partition, context)
        }
    }

    #[tokio::test]
    async fn execute_starts_independent_inputs_before_deciding_and_scans_once()
    -> Result<()> {
        let context = context()?;
        for threshold in [0, usize::MAX] {
            let original = plan(&context, "SELECT * FROM l JOIN r ON l.k = r.k").await?;
            let build_starts = Arc::new(AtomicUsize::new(0));
            let probe_starts = Arc::new(AtomicUsize::new(0));
            let mut hash_join = None;
            original.apply(|node| {
                if node.downcast_ref::<HashJoinExec>().is_some() {
                    hash_join = Some(Arc::clone(node));
                    return Ok(TreeNodeRecursion::Stop);
                }
                Ok(TreeNodeRecursion::Continue)
            })?;
            let hash_join = hash_join.expect("fixture must contain a hash join");
            let join = hash_join.downcast_ref::<HashJoinExec>().unwrap();
            assert_eq!(join.left().output_partitioning().partition_count(), 1);
            let tracked = join
                .builder()
                .with_new_children(vec![
                    Arc::new(StartsExec {
                        input: Arc::clone(join.left()),
                        starts: Arc::clone(&build_starts),
                    }),
                    Arc::new(StartsExec {
                        input: Arc::clone(join.right()),
                        starts: Arc::clone(&probe_starts),
                    }),
                ])?
                .recompute_properties()
                .build_exec()?;
            // Isolate the join's execute contract from parent operators that
            // defer execute and installation repair that moves coalescing.
            let plan: Arc<dyn ExecutionPlan> =
                Arc::new(AdaptiveJoinExec::new(tracked, threshold)?);
            drop(hash_join);
            drop(original);
            let count = plan.output_partitioning().partition_count();
            let outputs = (0..count)
                .map(|partition| plan.execute(partition, context.task_ctx()))
                .collect::<Result<Vec<_>>>()?;
            assert_eq!(build_starts.load(Ordering::SeqCst), 1);
            assert_eq!(probe_starts.load(Ordering::SeqCst), count);
            assert!(
                plan.downcast_ref::<AdaptiveJoinExec>()
                    .unwrap()
                    .decision
                    .lock()
                    .unwrap()
                    .as_ref()
                    .unwrap()
                    .peek()
                    .is_none()
            );
            let batches = try_join_all(
                outputs
                    .into_iter()
                    .map(|stream| stream.try_collect::<Vec<_>>()),
            )
            .await?;
            assert_eq!(
                batches
                    .iter()
                    .flatten()
                    .map(RecordBatch::num_rows)
                    .sum::<usize>(),
                5
            );
            assert_eq!(build_starts.load(Ordering::SeqCst), 1);
            assert_eq!(probe_starts.load(Ordering::SeqCst), count);
            assert_mode(&[Arc::clone(&plan)], expected_mode(threshold));
            drop(plan);
            assert_eq!(context.task_ctx().memory_pool().reserved(), 0);
        }
        Ok(())
    }
    #[tokio::test]
    async fn materialization_reports_memory_errors_and_releases_reservations()
    -> Result<()> {
        let context = context()?;
        let plan = context.table("l").await?.create_physical_plan().await?;
        let pool = Arc::new(GreedyMemoryPool::new(1));
        let runtime = Arc::new(RuntimeEnvBuilder::new().with_memory_pool(pool).build()?);
        let task_context = Arc::new(TaskContext::default().with_runtime(runtime));
        let boundary = BuildStageBoundary::new(plan);
        for partition in 0..boundary.partitions.len() {
            boundary.prime(partition, Arc::clone(&task_context))?;
            boundary.prime(partition, Arc::clone(&task_context))?;
        }
        boundary.wait_ready().await;
        assert!(boundary.build_bytes().is_err());
        drop(boundary);
        assert_eq!(task_context.memory_pool().reserved(), 0);
        Ok(())
    }

    #[tokio::test]
    async fn boundary_blocks_until_release_and_frees_cancelled_output() -> Result<()> {
        let context = context()?;
        let plan = context.table("l").await?.create_physical_plan().await?;
        let boundary = BuildStageBoundary::new(plan);
        let task_context = context.task_ctx();
        for partition in 0..boundary.partitions.len() {
            boundary.prime(partition, Arc::clone(&task_context))?;
        }
        boundary.wait_ready().await;
        assert!(boundary.build_bytes()? > 0);
        assert!(task_context.memory_pool().reserved() > 0);
        let mut output = boundary.execute(0, Arc::clone(&task_context))?;
        assert!(output.next().now_or_never().is_none());
        boundary.release();
        assert_eq!(output.try_next().await?.unwrap().num_rows(), 3);
        drop(output);
        drop(boundary);
        assert_eq!(task_context.memory_pool().reserved(), 0);
        Ok(())
    }
}
