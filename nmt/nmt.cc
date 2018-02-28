/* Copyright 2018 Stanford, NVIDIA
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstdio>
#include "rnn.h"
#include "rnn_mapper.h"

using namespace Legion;

LegionRuntime::Logger::Category log_nmt("nmt");

void parse_input_args(char **argv, int argc,
                      int &batch_size, int &num_layers, int &seq_length,
                      int &hidden_size, int &embed_size);
void set_global_config(GlobalConfig &global, int num_layers,
                       int seq_length, int num_parts);

void top_level_task(const Task *task, const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  int batch_size = 512;
  int num_layers = 2;
  int seq_length = 40;
  int hidden_size = 1024;
  int embed_size = 1024;
  int vocab_size = 32 * 1024;
  int num_nodes = 1;
  int workers_per_node = 4;
  int num_parts = 4;
  int num_iterations = 10;
  {
    const InputArgs &command_args = HighLevelRuntime::get_input_args();
    char **argv = command_args.argv;
    int argc = command_args.argc;
    parse_input_args(argv, argc, batch_size, num_layers, seq_length,
                     hidden_size, embed_size);
  }
  GlobalConfig global;
  set_global_config(global, num_layers, seq_length, num_parts);
  RnnModel model(batch_size, num_layers, seq_length, hidden_size, embed_size,
                 vocab_size, num_parts, num_nodes, workers_per_node,
                 global, ctx, runtime);
  ArgumentMap local_args;
  size_t workSpaceSize = (size_t) 2 * 1024 * 1024 * 1024;
  Rect<1> workers_rect(Point<1>(0), Point<1>(num_nodes * workers_per_node-1));
  int idx = 0;
  for (PointInRectIterator<1> it(workers_rect); it(); it++) {
    TaskLauncher launcher(CUDNN_INIT_TASK_ID,
                          TaskArgument(&workSpaceSize, sizeof(workSpaceSize)),
                          Predicate::TRUE_PRED, 0/*MapperID*/,
                          RnnMapper::assign_to_gpu(idx));
    Future f = runtime->execute_task(ctx, launcher);
    model.dnn_handlers[idx++] = f.get_result<DnnHandle>();
  }

  model.init();
  double ts_start = Realm::Clock::current_time_in_microseconds();
  for (int i = 0; i < num_iterations; i++) {
    model.forward();
    model.backward();
    model.update();
  }
  runtime->issue_execution_fence(ctx);
  TimingLauncher timer(MEASURE_MICRO_SECONDS);
  Future future = runtime->issue_timing_measurement(ctx, timer);
  future.get_void_result();
  double ts_end = Realm::Clock::current_time_in_microseconds();
  double run_time = 1e-6 * (ts_end - ts_start);
  printf("time = %.4fs\n", run_time);
}

int main(int argc, char **argv)
{
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  // DNN_INIT_TASK
  {
    TaskVariantRegistrar registrar(CUDNN_INIT_TASK_ID, "cudnn_init_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DnnHandle, init_cudnn>(registrar, "cudnn_init_task");
  }
  // LSTM task
  {
    TaskVariantRegistrar registrar(LSTM_INIT_TASK_ID, "lstm_init_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta*, LSTM::init_task>(registrar, "lstm_init_task");
  }
  {
    TaskVariantRegistrar registrar(LSTM_FWD_TASK_ID, "lstm_fwd_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<LSTM::forward_task>(registrar, "lstm_fwd_task");
  }
  {
    TaskVariantRegistrar registrar(LSTM_BWD_TASK_ID, "lstm_bwd_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<LSTM::backward_task>(registrar, "lstm_bwd_task");
  }
  // Rnn Linear task
  {
    TaskVariantRegistrar registrar(RNN_LINEAR_INIT_TASK_ID, "linear_init_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta*, Linear::init_task>(registrar, "linear_init_task");
  }
  {
    TaskVariantRegistrar registrar(RNN_LINEAR_FWD_TASK_ID, "linar_fwd_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Linear::forward_task>(registrar, "linear_fwd_task");
  }
  {
    TaskVariantRegistrar registrar(RNN_LINEAR_BWD_TASK_ID, "linear_bwd_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Linear::backward_task>(registrar, "linear_bwd_task");
  }
  {
    TaskVariantRegistrar registrar(RNN_LINEAR_BWD2_TASK_ID, "linear_bwd2_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Linear::backward2_task>(registrar, "linear_bwd2_task");
  }

  // Params update task
  {
    TaskVariantRegistrar registrar(PARAMS_UPD_TASK_ID, "params_upd_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<RnnModel::params_update_task>(registrar, "params_upd_task");
  }

  Runtime::add_registration_callback(update_mappers);
  return Runtime::start(argc, argv);
}

void parse_input_args(char **argv, int argc,
                      int &batch_size, int &num_layers, int &seq_length,
                      int &hidden_size, int &embed_size)
{
  for (int i = 1; i < argc; i++)
  {
    if (!strcmp(argv[i], "-b"))
    {
      batch_size = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "-l"))
    {
      num_layers = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "-s"))
    {
      seq_length = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "-h"))
    {
      hidden_size = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "-e"))
    {
      embed_size = atoi(argv[++i]);
      continue;
    }
  }
}

void set_global_config(GlobalConfig &global, int num_layers, int seq_length, int num_parts)
{
  for (int i = 0; i < num_layers; i++)
    for (int j = 0; j * LSTM_PER_NODE_LENGTH < 2 * seq_length; j++) {
      ParallelConfig pc;
      pc.nDims = 1;
      pc.dim[0] = num_parts;
      for (int k = 0; k < num_parts; k++)
        pc.gpu[k] = k;
      global.lstm[i][j] = pc;
    }
  for (int i = 0; i * LSTM_PER_NODE_LENGTH < seq_length; i++) {
    ParallelConfig pc;
    pc.nDims = 2;
    pc.dim[0] = num_parts;
    pc.dim[1] = 1;
    for (int j = 0; j < num_parts; j++)
      pc.gpu[j] = j;
    global.linear[i] = pc;
  }
}

