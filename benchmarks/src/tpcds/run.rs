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

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use crate::adaptive_join::install_adaptive_joins;
use crate::util::{BenchmarkRun, CommonOpt, QueryResult, print_memory_stats};

use arrow::datatypes::Schema;
use arrow::record_batch::RecordBatch;
use arrow::util::pretty::{self, pretty_format_batches};
use datafusion::datasource::file_format::parquet::ParquetFormat;
use datafusion::datasource::listing::{
    ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl,
};
use datafusion::datasource::{MemTable, TableProvider};
use datafusion::error::Result;
use datafusion::execution::SessionStateBuilder;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::display::DisplayableExecutionPlan;
use datafusion::physical_plan::joins::{HashJoinExec, PartitionMode};
use datafusion::physical_plan::{collect, displayable};
use datafusion::prelude::*;
use datafusion_common::config::ConfigOptions;
use datafusion_common::instant::Instant;
use datafusion_common::tree_node::{Transformed, TreeNode};
use datafusion_common::utils::get_available_parallelism;
use datafusion_common::{Constraint, Constraints, DEFAULT_PARQUET_EXTENSION, plan_err};

use clap::Args;
use log::info;

// hack to avoid `default_value is meaningless for bool` errors
type BoolDefaultTrue = bool;
pub const TPCDS_QUERY_START_ID: usize = 1;
pub const TPCDS_QUERY_END_ID: usize = 99;

pub const TPCDS_TABLES: &[&str] = &[
    "call_center",
    "customer_address",
    "household_demographics",
    "promotion",
    "store_sales",
    "web_page",
    "catalog_page",
    "customer_demographics",
    "income_band",
    "reason",
    "store",
    "web_returns",
    "catalog_returns",
    "customer",
    "inventory",
    "ship_mode",
    "time_dim",
    "web_sales",
    "catalog_sales",
    "date_dim",
    "item",
    "store_returns",
    "warehouse",
    "web_site",
];

static TPCDS_PRIMARY_KEYS: &[(&str, &[&str])] = &[
    ("call_center", &["cc_call_center_sk"]),
    ("catalog_page", &["cp_catalog_page_sk"]),
    ("catalog_returns", &["cr_item_sk", "cr_order_number"]),
    ("catalog_sales", &["cs_item_sk", "cs_order_number"]),
    ("customer", &["c_customer_sk"]),
    ("customer_address", &["ca_address_sk"]),
    ("customer_demographics", &["cd_demo_sk"]),
    ("date_dim", &["d_date_sk"]),
    ("household_demographics", &["hd_demo_sk"]),
    ("income_band", &["ib_income_band_sk"]),
    (
        "inventory",
        &["inv_date_sk", "inv_item_sk", "inv_warehouse_sk"],
    ),
    ("item", &["i_item_sk"]),
    ("promotion", &["p_promo_sk"]),
    ("reason", &["r_reason_sk"]),
    ("ship_mode", &["sm_ship_mode_sk"]),
    ("store", &["s_store_sk"]),
    ("store_returns", &["sr_item_sk", "sr_ticket_number"]),
    ("store_sales", &["ss_item_sk", "ss_ticket_number"]),
    ("time_dim", &["t_time_sk"]),
    ("warehouse", &["w_warehouse_sk"]),
    ("web_page", &["wp_web_page_sk"]),
    ("web_returns", &["wr_item_sk", "wr_order_number"]),
    ("web_sales", &["ws_item_sk", "ws_order_number"]),
    ("web_site", &["web_site_sk"]),
];

/// Get the constraints for a TPC-DS table. Only primary keys are returned;
/// TPC-DS also defines foreign keys, but those are currently unsupported.
fn table_constraints(table: &str, schema: &Schema) -> Constraints {
    let columns = TPCDS_PRIMARY_KEYS
        .iter()
        .find(|(name, _)| *name == table)
        .map(|(_, columns)| *columns)
        .unwrap_or_else(|| unimplemented!("unknown TPC-DS table: {table}"));

    Constraints::new_unverified(vec![primary_key(schema, columns)])
}

fn primary_key(schema: &Schema, column_names: &[&str]) -> Constraint {
    let indices = column_names
        .iter()
        .map(|column_name| {
            schema.index_of(column_name).unwrap_or_else(|_| {
                panic!("primary key column '{column_name}' not found in schema")
            })
        })
        .collect();

    Constraint::PrimaryKey(indices)
}

/// Get the SQL statements from the specified query file
pub fn get_query_sql(base_query_path: &str, query: usize) -> Result<Vec<String>> {
    if query > 0 && query < 100 {
        let filename = format!("{base_query_path}/{query}.sql");
        let mut errors = vec![];
        match fs::read_to_string(&filename) {
            Ok(contents) => {
                return Ok(contents
                    .split(';')
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string())
                    .collect());
            }
            Err(e) => errors.push(format!("{filename}: {e}")),
        }

        plan_err!("invalid query. Could not find query: {:?}", errors)
    } else {
        plan_err!("invalid query. Expected value between 1 and 99")
    }
}

/// Run the tpcds benchmark.
#[derive(Debug, Args, Clone)]
#[command(verbatim_doc_comment)]
pub struct RunOpt {
    /// Query number. If not specified, runs all queries
    #[arg(short, long)]
    pub query: Option<usize>,

    /// Common options
    #[command(flatten)]
    common: CommonOpt,

    /// Path to data files
    #[arg(required = true, short = 'p', long = "path")]
    path: PathBuf,

    /// Path to query files
    #[arg(required = true, short = 'Q', long = "query_path")]
    query_path: PathBuf,

    /// Load the data into a MemTable before executing the query
    #[arg(short = 'm', long = "mem-table")]
    mem_table: bool,

    /// Path to machine readable output file
    #[arg(short = 'o', long = "output")]
    output_path: Option<PathBuf>,

    /// Whether to disable collection of statistics (and cost based optimizations) or not.
    #[arg(short = 'S', long = "disable-statistics")]
    disable_statistics: bool,

    /// If true then hash join used, if false then sort merge join
    /// True by default.
    #[arg(short = 'j', long = "prefer_hash_join", default_value = "true")]
    prefer_hash_join: BoolDefaultTrue,

    /// If true then Piecewise Merge Join can be used, if false then it will opt for Nested Loop Join
    /// False by default.
    #[arg(
        short = 'w',
        long = "enable_piecewise_merge_join",
        default_value = "false"
    )]
    enable_piecewise_merge_join: BoolDefaultTrue,

    /// Mark the first column of each table as sorted in ascending order.
    /// The tables should have been created with the `--sort` option for this to have any effect.
    #[arg(short = 't', long = "sorted")]
    sorted: bool,

    /// How many bytes to buffer on the probe side of hash joins.
    #[arg(long, default_value = "0")]
    hash_join_buffering_capacity: usize,

    /// Disable this draft's StageBoundary join experiment and use the default planner.
    #[arg(long, env = "DFBENCH_DISABLE_ADAPTIVE_JOINS")]
    disable_adaptive_joins: bool,

    /// Experiment: partition when total buffered build Arrow array memory reaches
    /// this many bytes. Includes array overhead; shared buffers may count repeatedly.
    #[arg(
        long,
        env = "DFBENCH_ADAPTIVE_JOIN_THRESHOLD_BYTES",
        default_value = "2621440"
    )]
    adaptive_join_threshold_bytes: usize,
}

impl RunOpt {
    pub async fn run(self) -> Result<()> {
        println!("Running benchmarks with the following options: {self:?}");
        let query_range = match self.query {
            Some(query_id) => query_id..=query_id,
            None => TPCDS_QUERY_START_ID..=TPCDS_QUERY_END_ID,
        };

        let mut benchmark_run = BenchmarkRun::new();
        let mut config = self
            .common
            .config()?
            .with_collect_statistics(!self.disable_statistics);
        config.options_mut().optimizer.prefer_hash_join = self.prefer_hash_join;
        config.options_mut().optimizer.enable_piecewise_merge_join =
            self.enable_piecewise_merge_join;
        config.options_mut().execution.hash_join_buffering_capacity =
            self.hash_join_buffering_capacity;
        let rt = self.common.build_runtime()?;
        let ctx = SessionContext::new_with_config_rt(config, rt);
        let ctx = if self.disable_adaptive_joins {
            ctx
        } else {
            collect_left_context(&ctx)
        };
        benchmark_run.set_memory_pool(&ctx.runtime_env().memory_pool);
        // register tables
        self.register_tables(&ctx).await?;

        for query_id in query_range {
            benchmark_run.start_new_case(&format!("Query {query_id}"));
            let query_run = self.benchmark_query(query_id, &ctx).await;
            match query_run {
                Ok(query_results) => {
                    for iter in query_results {
                        benchmark_run.write_iter(iter.elapsed, iter.row_count);
                    }
                }
                Err(e) => {
                    benchmark_run.mark_failed();
                    eprintln!("Query {query_id} failed: {e}");
                }
            }
        }
        benchmark_run.maybe_write_json(self.output_path.as_ref())?;
        benchmark_run.maybe_print_failures();
        Ok(())
    }

    async fn benchmark_query(
        &self,
        query_id: usize,
        ctx: &SessionContext,
    ) -> Result<Vec<QueryResult>> {
        let mut millis = vec![];
        // run benchmark
        let mut query_results = vec![];

        let sql = &get_query_sql(self.query_path.to_str().unwrap(), query_id)?;

        if self.common.debug {
            println!("=== SQL for query {query_id} ===\n{}\n", sql.join(";\n"));
        }

        for i in 0..self.iterations() {
            let start = Instant::now();

            // query 15 is special, with 3 statements. the second statement is the one from which we
            // want to capture the results
            let mut result = vec![];

            for query in sql {
                result = self.execute_query(ctx, query).await?;
            }

            let elapsed = start.elapsed();
            let ms = elapsed.as_secs_f64() * 1000.0;
            millis.push(ms);
            info!("output:\n\n{}\n\n", pretty_format_batches(&result)?);
            let row_count = result.iter().map(|b| b.num_rows()).sum();
            println!(
                "Query {query_id} iteration {i} took {ms:.1} ms and returned {row_count} rows"
            );
            query_results.push(QueryResult { elapsed, row_count });
        }

        let avg = millis.iter().sum::<f64>() / millis.len() as f64;
        println!("Query {query_id} avg time: {avg:.2} ms");

        // Print memory stats using mimalloc (only when compiled with --features mimalloc_extended)
        print_memory_stats(&*ctx.runtime_env().memory_pool);

        Ok(query_results)
    }

    async fn register_tables(&self, ctx: &SessionContext) -> Result<()> {
        for table in TPCDS_TABLES {
            let table_provider = { self.get_table(ctx, table).await? };

            if self.mem_table {
                println!("Loading table '{table}' into memory");
                let start = Instant::now();
                let memtable =
                    MemTable::load(table_provider, Some(self.partitions()), &ctx.state())
                        .await?;
                println!(
                    "Loaded table '{}' into memory in {} ms",
                    table,
                    start.elapsed().as_millis()
                );
                ctx.register_table(*table, Arc::new(memtable))?;
            } else {
                ctx.register_table(*table, table_provider)?;
            }
        }
        Ok(())
    }

    async fn execute_query(
        &self,
        ctx: &SessionContext,
        sql: &str,
    ) -> Result<Vec<RecordBatch>> {
        let debug = self.common.debug;
        let plan = ctx.sql(sql).await?;
        let (state, plan) = plan.into_parts();

        if debug {
            println!("=== Logical plan ===\n{plan}\n");
        }

        let plan = state.optimize(&plan)?;
        if debug {
            println!("=== Optimized logical plan ===\n{plan}\n");
        }
        let physical_plan = state.create_physical_plan(&plan).await?;
        let physical_plan = if self.disable_adaptive_joins {
            physical_plan
        } else {
            install_adaptive_joins(
                physical_plan,
                &state.task_ctx(),
                self.adaptive_join_threshold_bytes,
            )?
        };
        if debug {
            println!(
                "=== Physical plan ===\n{}\n",
                displayable(physical_plan.as_ref()).indent(true)
            );
        }
        let result = collect(physical_plan.clone(), state.task_ctx()).await?;
        if debug {
            println!(
                "=== Physical plan with metrics ===\n{}\n",
                DisplayableExecutionPlan::with_metrics(physical_plan.as_ref())
                    .indent(true)
            );
            if !result.is_empty() {
                // do not call print_batches if there are no batches as the result is confusing
                // and makes it look like there is a batch with no columns
                pretty::print_batches(&result)?;
            }
        }
        Ok(result)
    }

    async fn get_table(
        &self,
        ctx: &SessionContext,
        table: &str,
    ) -> Result<Arc<dyn TableProvider>> {
        let path = self.path.to_str().unwrap();

        // Obtain a snapshot of the SessionState
        let state = ctx.state();
        let path = format!("{path}/{table}.parquet");

        // Check if the file exists
        if !std::path::Path::new(&path).exists() {
            eprintln!("Warning registering {table}: Table file does not exist: {path}");
        }

        let format = ParquetFormat::default()
            .with_options(ctx.state().table_options().parquet.clone());

        let table_path = ListingTableUrl::parse(path)?;
        let options = ListingOptions::new(Arc::new(format))
            .with_file_extension(DEFAULT_PARQUET_EXTENSION);

        let schema = options.infer_schema(&state, &table_path).await?;
        let constraints = table_constraints(table, schema.as_ref());

        if self.common.debug {
            println!(
                "Inferred schema from {table_path} for table '{table}':\n{schema:#?}\n"
            );
        }

        let options = if self.sorted {
            let key_column_name = schema.fields()[0].name();
            options
                .with_file_sort_order(vec![vec![col(key_column_name).sort(true, false)]])
        } else {
            options
        };

        let config = ListingTableConfig::new(table_path)
            .with_listing_options(options)
            .with_schema(schema);

        let provider = ListingTable::try_new(config)?
            .with_constraints(constraints)
            .with_cache(ctx.runtime_env().cache_manager.get_file_statistic_cache());

        Ok(Arc::new(provider))
    }

    fn iterations(&self) -> usize {
        self.common.iterations
    }

    fn partitions(&self) -> usize {
        self.common
            .partitions
            .unwrap_or_else(get_available_parallelism)
    }
}

fn collect_left_context(ctx: &SessionContext) -> SessionContext {
    let state = ctx.state();
    let mut rules = state.physical_optimizers().to_vec();
    let selection = rules
        .iter()
        .position(|rule| rule.name() == "join_selection")
        .expect("default optimizer includes join selection");
    // Preserve the default build side, before distribution requirements add
    // join-specific probe exchanges.
    rules.insert(selection + 1, Arc::new(ForceCollectLeft));
    SessionContext::new_with_state(
        SessionStateBuilder::from(state)
            .with_physical_optimizer_rules(rules)
            .build(),
    )
}

#[derive(Debug)]
struct ForceCollectLeft;

impl PhysicalOptimizerRule for ForceCollectLeft {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(plan
            .transform_up(|plan| {
                let Some(join) = plan.downcast_ref::<HashJoinExec>() else {
                    return Ok(Transformed::no(plan));
                };
                if *join.partition_mode() == PartitionMode::CollectLeft {
                    return Ok(Transformed::no(plan));
                }
                Ok(Transformed::yes(
                    join.builder()
                        .with_partition_mode(PartitionMode::CollectLeft)
                        .recompute_properties()
                        .build_exec()?,
                ))
            })?
            .data)
    }

    fn name(&self) -> &str {
        "benchmark_force_collect_left"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod adaptive_planner_tests {
    use super::*;
    use arrow::array::{ArrayRef, Int32Array};
    use datafusion::physical_plan::ExecutionPlanProperties;
    use datafusion_common::tree_node::TreeNodeRecursion;

    #[tokio::test]
    async fn forced_collect_left_precedes_distribution_enforcement() -> Result<()> {
        let mut config = SessionConfig::new().with_target_partitions(4);
        config
            .options_mut()
            .optimizer
            .hash_join_single_partition_threshold = 0;
        config
            .options_mut()
            .optimizer
            .hash_join_single_partition_threshold_rows = 0;
        let ctx = SessionContext::new_with_config(config);
        for table in ["l", "r"] {
            let batch = RecordBatch::try_from_iter(vec![(
                "k",
                Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef,
            )])?;
            ctx.register_batch(table, batch)?;
        }
        let sql = "SELECT l.k FROM l JOIN r ON l.k = r.k ORDER BY l.k";
        let baseline = ctx.sql(sql).await?.create_physical_plan().await?;
        let mut modes = vec![];
        baseline.apply(|node| {
            if let Some(join) = node.downcast_ref::<HashJoinExec>() {
                modes.push(*join.partition_mode());
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
        assert_eq!(modes, vec![PartitionMode::Partitioned]);
        let expected = collect(baseline, ctx.task_ctx()).await?;
        let ctx = collect_left_context(&ctx);
        let forced = ctx.sql(sql).await?.create_physical_plan().await?;
        let mut modes = vec![];
        forced.apply(|node| {
            if let Some(join) = node.downcast_ref::<HashJoinExec>() {
                modes.push(*join.partition_mode());
                assert_eq!(join.left().output_partitioning().partition_count(), 1);
                assert!(!matches!(
                    join.right().output_partitioning(),
                    datafusion::physical_plan::Partitioning::Hash(_, _)
                ));
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
        assert_eq!(modes, vec![PartitionMode::CollectLeft]);
        assert_eq!(expected, collect(forced, ctx.task_ctx()).await?);
        Ok(())
    }
}
