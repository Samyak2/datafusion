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

//! Defines the merge plan for executing partitions in parallel and then merging the results
//! into a single partition

use std::sync::Arc;

use super::metrics::{BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet};
use super::stream::{ObservedStream, RecordBatchReceiverStream};
use super::{
    DisplayAs, ExecutionPlanProperties, PlanProperties, SendableRecordBatchStream,
    Statistics,
};
use crate::execution_plan::{CardinalityEffect, EvaluationType, SchedulingType};
use crate::filter_pushdown::{FilterDescription, FilterPushdownPhase};
use crate::projection::{ProjectionExec, make_with_child};
use crate::sort_pushdown::SortOrderPushdownResult;
use crate::{DisplayFormatType, ExecutionPlan, Partitioning, check_if_same_properties};
use datafusion_physical_expr_common::sort_expr::PhysicalSortExpr;

use datafusion_common::config::ConfigOptions;
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_common::{Result, assert_eq_or_internal_err, internal_err};
use datafusion_execution::TaskContext;
use datafusion_physical_expr::PhysicalExpr;

/// Merge execution plan executes partitions in parallel and combines them into a single
/// partition. No guarantees are made about the order of the resulting partition.
#[derive(Debug, Clone)]
pub struct CoalescePartitionsExec {
    /// Input execution plan
    input: Arc<dyn ExecutionPlan>,
    /// Execution metrics
    metrics: ExecutionPlanMetricsSet,
    cache: Arc<PlanProperties>,
    /// Optional number of rows to fetch. Stops producing rows after this fetch
    pub(crate) fetch: Option<usize>,
}

impl CoalescePartitionsExec {
    /// Create a new CoalescePartitionsExec
    pub fn new(input: Arc<dyn ExecutionPlan>) -> Self {
        let cache = Self::compute_properties(&input);
        CoalescePartitionsExec {
            input,
            metrics: ExecutionPlanMetricsSet::new(),
            cache: Arc::new(cache),
            fetch: None,
        }
    }

    /// Update fetch with the argument
    pub fn with_fetch(mut self, fetch: Option<usize>) -> Self {
        self.fetch = fetch;
        self
    }

    /// Input execution plan
    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }

    /// This function creates the cache object that stores the plan properties such as schema, equivalence properties, ordering, partitioning, etc.
    fn compute_properties(input: &Arc<dyn ExecutionPlan>) -> PlanProperties {
        let input_partitions = input.output_partitioning().partition_count();
        let (drive, scheduling) = if input_partitions > 1 {
            (EvaluationType::Eager, SchedulingType::Cooperative)
        } else {
            (
                input.properties().evaluation_type,
                input.properties().scheduling_type,
            )
        };

        // Coalescing partitions loses existing orderings:
        let mut eq_properties = input.equivalence_properties().clone();
        eq_properties.clear_orderings();
        eq_properties.clear_per_partition_constants();
        PlanProperties::new(
            eq_properties,                        // Equivalence Properties
            Partitioning::UnknownPartitioning(1), // Output Partitioning
            input.pipeline_behavior(),
            input.boundedness(),
        )
        .with_evaluation_type(drive)
        .with_scheduling_type(scheduling)
    }

    fn with_new_children_and_same_properties(
        &self,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Self {
        Self {
            input: children.swap_remove(0),
            metrics: ExecutionPlanMetricsSet::new(),
            ..Self::clone(self)
        }
    }
}

impl DisplayAs for CoalescePartitionsExec {
    fn fmt_as(
        &self,
        t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => match self.fetch {
                Some(fetch) => {
                    write!(f, "CoalescePartitionsExec: fetch={fetch}")
                }
                None => write!(f, "CoalescePartitionsExec"),
            },
            DisplayFormatType::TreeRender => match self.fetch {
                Some(fetch) => {
                    write!(f, "limit: {fetch}")
                }
                None => write!(f, ""),
            },
        }
    }
}

impl ExecutionPlan for CoalescePartitionsExec {
    fn name(&self) -> &'static str {
        "CoalescePartitionsExec"
    }

    /// Return a reference to Any that can be used for downcasting
    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&dyn PhysicalExpr) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        check_if_same_properties!(self, children);
        let mut plan = CoalescePartitionsExec::new(children.swap_remove(0));
        plan.fetch = self.fetch;
        Ok(Arc::new(plan))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        // CoalescePartitionsExec produces a single partition
        assert_eq_or_internal_err!(
            partition,
            0,
            "CoalescePartitionsExec invalid partition {partition}"
        );

        let input_partitions = self.input.output_partitioning().partition_count();
        match input_partitions {
            0 => internal_err!(
                "CoalescePartitionsExec requires at least one input partition"
            ),
            1 => {
                // single-partition path: execute child directly, but ensure fetch is respected
                // (wrap with ObservedStream only if fetch is present so we don't add overhead otherwise)
                let child_stream = self.input.execute(0, context)?;
                if self.fetch.is_some() {
                    let baseline_metrics = BaselineMetrics::new(&self.metrics, partition);
                    return Ok(Box::pin(ObservedStream::new(
                        child_stream,
                        baseline_metrics,
                        self.fetch,
                    )));
                }
                Ok(child_stream)
            }
            _ => {
                let baseline_metrics = BaselineMetrics::new(&self.metrics, partition);
                // record the (very) minimal work done so that
                // elapsed_compute is not reported as 0
                let elapsed_compute = baseline_metrics.elapsed_compute().clone();
                let _timer = elapsed_compute.timer();

                // use a stream that allows each sender to put in at
                // least one result in an attempt to maximize
                // parallelism.
                let mut builder =
                    RecordBatchReceiverStream::builder(self.schema(), input_partitions);

                // spawn independent tasks whose resulting streams (of batches)
                // are sent to the channel for consumption.
                for part_i in 0..input_partitions {
                    builder.run_input(
                        Arc::clone(&self.input),
                        part_i,
                        Arc::clone(&context),
                    );
                }

                let stream = builder.build();
                Ok(Box::pin(ObservedStream::new(
                    stream,
                    baseline_metrics,
                    self.fetch,
                )))
            }
        }
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn partition_statistics(&self, _partition: Option<usize>) -> Result<Arc<Statistics>> {
        let stats = Arc::unwrap_or_clone(self.input.partition_statistics(None)?);
        Ok(Arc::new(stats.with_fetch(self.fetch, 0, 1)?))
    }

    fn supports_limit_pushdown(&self) -> bool {
        true
    }

    fn cardinality_effect(&self) -> CardinalityEffect {
        CardinalityEffect::Equal
    }

    /// Tries to swap `projection` with its input, which is known to be a
    /// [`CoalescePartitionsExec`]. If possible, performs the swap and returns
    /// [`CoalescePartitionsExec`] as the top plan. Otherwise, returns `None`.
    fn try_swapping_with_projection(
        &self,
        projection: &ProjectionExec,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        // If the projection does not narrow the schema, we should not try to push it down:
        if projection.expr().len() >= projection.input().schema().fields().len() {
            return Ok(None);
        }
        // CoalescePartitionsExec always has a single child, so zero indexing is safe.
        make_with_child(projection, projection.input().children()[0]).map(|e| {
            if self.fetch.is_some() {
                let mut plan = CoalescePartitionsExec::new(e);
                plan.fetch = self.fetch;
                Some(Arc::new(plan) as _)
            } else {
                Some(Arc::new(CoalescePartitionsExec::new(e)) as _)
            }
        })
    }

    fn fetch(&self) -> Option<usize> {
        self.fetch
    }

    fn with_fetch(&self, limit: Option<usize>) -> Option<Arc<dyn ExecutionPlan>> {
        Some(Arc::new(CoalescePartitionsExec {
            input: Arc::clone(&self.input),
            fetch: limit,
            metrics: self.metrics.clone(),
            cache: Arc::clone(&self.cache),
        }))
    }

    fn with_preserve_order(
        &self,
        preserve_order: bool,
    ) -> Option<Arc<dyn ExecutionPlan>> {
        self.input
            .with_preserve_order(preserve_order)
            .and_then(|new_input| {
                Arc::new(self.clone())
                    .with_new_children(vec![new_input])
                    .ok()
            })
    }

    fn gather_filters_for_pushdown(
        &self,
        _phase: FilterPushdownPhase,
        parent_filters: Vec<Arc<dyn PhysicalExpr>>,
        _config: &ConfigOptions,
    ) -> Result<FilterDescription> {
        FilterDescription::from_children(parent_filters, &self.children())
    }

    fn try_pushdown_sort(
        &self,
        order: &[PhysicalSortExpr],
    ) -> Result<SortOrderPushdownResult<Arc<dyn ExecutionPlan>>> {
        // CoalescePartitionsExec merges multiple partitions into one, which loses
        // global ordering. However, we can still push the sort requirement down
        // to optimize individual partitions - the Sort operator above will handle
        // the global ordering.
        //
        // Note: The result will always be at most Inexact (never Exact) when there
        // are multiple partitions, because merging destroys global ordering.
        let result = self.input.try_pushdown_sort(order)?;

        // If we have multiple partitions, we can't return Exact even if the
        // underlying source claims Exact - merging destroys global ordering
        let has_multiple_partitions =
            self.input.output_partitioning().partition_count() > 1;

        result
            .try_map(|new_input| {
                Ok(
                    Arc::new(
                        CoalescePartitionsExec::new(new_input).with_fetch(self.fetch),
                    ) as Arc<dyn ExecutionPlan>,
                )
            })
            .map(|r| {
                if has_multiple_partitions {
                    // Downgrade Exact to Inexact when merging multiple partitions
                    r.into_inexact()
                } else {
                    r
                }
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregates::{AggregateExec, AggregateMode, PhysicalGroupBy};
    use crate::expressions::col;
    use crate::memory::{LazyBatchGenerator, LazyMemoryExec};
    use crate::repartition::RepartitionExec;
    use crate::test::exec::{
        BlockingExec, PanicExec, assert_strong_count_converges_to_zero,
    };
    use crate::test::{self, assert_is_pending};
    use crate::{collect, common};

    use arrow::array::{ArrayRef, UInt64Array};
    use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
    use arrow::record_batch::RecordBatch;
    use datafusion_common::Result;
    use datafusion_functions_aggregate::count::count_udaf;
    use datafusion_physical_expr::aggregate::AggregateExprBuilder;

    use futures::FutureExt;
    use parking_lot::RwLock;
    use std::any::Any;
    use std::fmt;
    use std::sync::{Arc, Weak};
    use std::time::{Duration, Instant};

    #[tokio::test]
    async fn merge() -> Result<()> {
        let task_ctx = Arc::new(TaskContext::default());

        let num_partitions = 4;
        let csv = test::scan_partitioned(num_partitions);

        // input should have 4 partitions
        assert_eq!(csv.output_partitioning().partition_count(), num_partitions);

        let merge = CoalescePartitionsExec::new(csv);

        // output of CoalescePartitionsExec should have a single partition
        assert_eq!(
            merge.properties().output_partitioning().partition_count(),
            1
        );

        // the result should contain 4 batches (one per input partition)
        let iter = merge.execute(0, task_ctx)?;
        let batches = common::collect(iter).await?;
        assert_eq!(batches.len(), num_partitions);

        // there should be a total of 400 rows (100 per each partition)
        let row_count: usize = batches.iter().map(|batch| batch.num_rows()).sum();
        assert_eq!(row_count, 400);

        Ok(())
    }

    async fn wait_for_repartition_drop_times(
        refs: &[Weak<RepartitionExec>],
        start: Instant,
    ) -> Vec<Duration> {
        let mut drop_times = vec![None; refs.len()];
        tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                for (idx, refs) in refs.iter().enumerate() {
                    if drop_times[idx].is_none() && refs.strong_count() == 0 {
                        drop_times[idx] = Some(start.elapsed());
                    }
                }

                if drop_times.iter().all(Option::is_some) {
                    break;
                }

                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .unwrap();

        drop_times
            .into_iter()
            .map(|drop_time| drop_time.expect("all repartition refs dropped"))
            .collect()
    }

    #[derive(Debug, Clone)]
    struct CountingGenerator {
        schema: SchemaRef,
        partition: usize,
        next_batch: usize,
        max_batches: usize,
        rows_per_batch: usize,
    }

    impl CountingGenerator {
        fn new(
            schema: SchemaRef,
            partition: usize,
            max_batches: usize,
            rows_per_batch: usize,
        ) -> Self {
            Self {
                schema,
                partition,
                next_batch: 0,
                max_batches,
                rows_per_batch,
            }
        }
    }

    impl fmt::Display for CountingGenerator {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(
                f,
                "CountingGenerator: partition={}, max_batches={}, rows_per_batch={}",
                self.partition, self.max_batches, self.rows_per_batch
            )
        }
    }

    impl LazyBatchGenerator for CountingGenerator {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn generate_next_batch(&mut self) -> Result<Option<RecordBatch>> {
            if self.next_batch == self.max_batches {
                return Ok(None);
            }

            let start = ((self.partition * self.max_batches + self.next_batch)
                * self.rows_per_batch) as u64;
            self.next_batch += 1;

            let values =
                UInt64Array::from_iter_values(start..start + self.rows_per_batch as u64);

            Ok(Some(RecordBatch::try_new(
                Arc::clone(&self.schema),
                vec![Arc::new(values) as ArrayRef],
            )?))
        }

        fn reset_state(&self) -> Arc<RwLock<dyn LazyBatchGenerator>> {
            Arc::new(RwLock::new(Self {
                schema: Arc::clone(&self.schema),
                partition: self.partition,
                next_batch: 0,
                max_batches: self.max_batches,
                rows_per_batch: self.rows_per_batch,
            }))
        }
    }

    fn high_cardinality_partial_aggregate(
        input: Arc<dyn ExecutionPlan>,
        schema: &SchemaRef,
    ) -> Result<Arc<AggregateExec>> {
        let groups =
            PhysicalGroupBy::new_single(vec![(col("a", schema)?, "a".to_string())]);
        let count = AggregateExprBuilder::new(count_udaf(), vec![col("a", schema)?])
            .schema(Arc::clone(schema))
            .alias("COUNT(a)")
            .build()?;

        Ok(Arc::new(AggregateExec::try_new(
            AggregateMode::Partial,
            groups,
            vec![Arc::new(count)],
            vec![None],
            input,
            Arc::clone(schema),
        )?))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore = "temporary diagnostic reproducer for layered cancellation delay"]
    async fn cancellation_delay_coalesce_repartition() -> Result<()> {
        let task_ctx = Arc::new(TaskContext::default());
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::UInt64, true)]));
        let input_partitions = 2;
        let batches_per_input_partition = 8;
        let rows_per_batch = 128_000;
        let generators = (0..input_partitions)
            .map(|partition| {
                Arc::new(RwLock::new(CountingGenerator::new(
                    Arc::clone(&schema),
                    partition,
                    batches_per_input_partition,
                    rows_per_batch,
                ))) as Arc<RwLock<dyn LazyBatchGenerator>>
            })
            .collect();
        let input = Arc::new(LazyMemoryExec::try_new(Arc::clone(&schema), generators)?);
        let input_refs = Arc::downgrade(&input);
        let mut plan: Arc<dyn ExecutionPlan> =
            high_cardinality_partial_aggregate(input, &schema)?;

        let layers = 512;
        let output_partitions = 32;
        let mut repartition_refs = Vec::with_capacity(layers);

        for _ in 0..layers {
            let repartition = Arc::new(RepartitionExec::try_new(
                plan,
                Partitioning::RoundRobinBatch(output_partitions),
            )?);
            repartition_refs.push(Arc::downgrade(&repartition));

            plan = Arc::new(CoalescePartitionsExec::new(repartition));
        }

        let handle = tokio::spawn(collect(plan, task_ctx));
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(!handle.is_finished(), "query finished before cancellation");

        let start = Instant::now();
        handle.abort();

        let drop_times = wait_for_repartition_drop_times(&repartition_refs, start).await;
        let total_elapsed = start.elapsed();

        for (idx, elapsed) in drop_times.iter().enumerate().rev() {
            let layer_from_top = layers - idx;
            if layer_from_top != 1 && layer_from_top != layers && layer_from_top % 32 != 0
            {
                continue;
            }
            println!(
                "layer_from_top={layer_from_top} repartition_drop_elapsed_ms={}",
                elapsed.as_millis()
            );
        }
        println!(
            "layers={layers} output_partitions={output_partitions} input_rows_per_partition={} cancellation_elapsed_ms={}",
            batches_per_input_partition * rows_per_batch,
            total_elapsed.as_millis()
        );
        println!(
            "input_plan_strong_count_after_cancel={}",
            input_refs.strong_count()
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_drop_cancel() -> Result<()> {
        let task_ctx = Arc::new(TaskContext::default());
        let schema =
            Arc::new(Schema::new(vec![Field::new("a", DataType::Float32, true)]));

        let blocking_exec = Arc::new(BlockingExec::new(Arc::clone(&schema), 2));
        let refs = blocking_exec.refs();
        let coalesce_partitions_exec =
            Arc::new(CoalescePartitionsExec::new(blocking_exec));

        let fut = collect(coalesce_partitions_exec, task_ctx);
        let mut fut = fut.boxed();

        assert_is_pending(&mut fut);
        drop(fut);
        assert_strong_count_converges_to_zero(refs).await;

        Ok(())
    }

    #[tokio::test]
    #[should_panic(expected = "PanickingStream did panic")]
    async fn test_panic() {
        let task_ctx = Arc::new(TaskContext::default());
        let schema =
            Arc::new(Schema::new(vec![Field::new("a", DataType::Float32, true)]));

        let panicking_exec = Arc::new(PanicExec::new(Arc::clone(&schema), 2));
        let coalesce_partitions_exec =
            Arc::new(CoalescePartitionsExec::new(panicking_exec));

        collect(coalesce_partitions_exec, task_ctx).await.unwrap();
    }

    #[tokio::test]
    async fn test_single_partition_with_fetch() -> Result<()> {
        let task_ctx = Arc::new(TaskContext::default());

        // Use existing scan_partitioned with 1 partition (returns 100 rows per partition)
        let input = test::scan_partitioned(1);

        // Test with fetch=3
        let coalesce = CoalescePartitionsExec::new(input).with_fetch(Some(3));

        let stream = coalesce.execute(0, task_ctx)?;
        let batches = common::collect(stream).await?;

        let row_count: usize = batches.iter().map(|batch| batch.num_rows()).sum();
        assert_eq!(row_count, 3, "Should only return 3 rows due to fetch=3");

        Ok(())
    }

    #[tokio::test]
    async fn test_multi_partition_with_fetch_one() -> Result<()> {
        let task_ctx = Arc::new(TaskContext::default());

        // Create 4 partitions, each with 100 rows
        // This simulates the real-world scenario where each partition has data
        let input = test::scan_partitioned(4);

        // Test with fetch=1 (the original bug: was returning multiple rows instead of 1)
        let coalesce = CoalescePartitionsExec::new(input).with_fetch(Some(1));

        let stream = coalesce.execute(0, task_ctx)?;
        let batches = common::collect(stream).await?;

        let row_count: usize = batches.iter().map(|batch| batch.num_rows()).sum();
        assert_eq!(
            row_count, 1,
            "Should only return 1 row due to fetch=1, not one per partition"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_single_partition_without_fetch() -> Result<()> {
        let task_ctx = Arc::new(TaskContext::default());

        // Use scan_partitioned with 1 partition
        let input = test::scan_partitioned(1);

        // Test without fetch (should return all rows)
        let coalesce = CoalescePartitionsExec::new(input);

        let stream = coalesce.execute(0, task_ctx)?;
        let batches = common::collect(stream).await?;

        let row_count: usize = batches.iter().map(|batch| batch.num_rows()).sum();
        assert_eq!(
            row_count, 100,
            "Should return all 100 rows when fetch is None"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_single_partition_fetch_larger_than_batch() -> Result<()> {
        let task_ctx = Arc::new(TaskContext::default());

        // Use scan_partitioned with 1 partition (returns 100 rows)
        let input = test::scan_partitioned(1);

        // Test with fetch larger than available rows
        let coalesce = CoalescePartitionsExec::new(input).with_fetch(Some(200));

        let stream = coalesce.execute(0, task_ctx)?;
        let batches = common::collect(stream).await?;

        let row_count: usize = batches.iter().map(|batch| batch.num_rows()).sum();
        assert_eq!(
            row_count, 100,
            "Should return all available rows (100) when fetch (200) is larger"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_multi_partition_fetch_exact_match() -> Result<()> {
        let task_ctx = Arc::new(TaskContext::default());

        // Create 4 partitions, each with 100 rows
        let num_partitions = 4;
        let csv = test::scan_partitioned(num_partitions);

        // Test with fetch=400 (exactly all rows)
        let coalesce = CoalescePartitionsExec::new(csv).with_fetch(Some(400));

        let stream = coalesce.execute(0, task_ctx)?;
        let batches = common::collect(stream).await?;

        let row_count: usize = batches.iter().map(|batch| batch.num_rows()).sum();
        assert_eq!(row_count, 400, "Should return exactly 400 rows");

        Ok(())
    }
}
