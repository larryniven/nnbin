#include "la/la-cpu.h"
#include "autodiff/autodiff.h"
#include "ebt/ebt.h"
#include "util/speech.h"
#include "util/batch.h"
#include "util/util.h"
#include <fstream>
#include <vector>
#include "nn/lstm.h"
#include "nn/lstm-tensor-tree.h"
#include "nn/nn.h"
#include <random>
#include "nn/tensor-tree.h"
#include "nn/lstm-frame.h"
#include <algorithm>

struct learning_env {

    batch::scp input_scp;
    batch::scp target_scp;

    int layer;
    std::shared_ptr<tensor_tree::vertex> param;

    std::shared_ptr<tensor_tree::optimizer> opt;

    std::string output_param;
    std::string output_opt_data;

    int seed;
    double dropout;
    double step_size;
    double clip;

    int batch_size;
    std::vector<int> indices;

    std::default_random_engine gen;

    std::unordered_map<std::string, std::string> args;

    learning_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "enhance-lstm-learn",
        "Train a LSTM frame classifier",
        {
            {"input-scp", "", true},
            {"target-scp", "", true},
            {"param", "", true},
            {"opt-data", "", true},
            {"output-param", "", false},
            {"output-opt-data", "", false},
            {"dropout", "", false},
            {"shuffle", "", false},
            {"seed", "", false},
            {"batch-size", "", false},
            {"opt", "const-step,rmsprop,adagrad", true},
            {"step-size", "", true},
            {"clip", "", false},
            {"decay", "", false},
        }
    };

    if (argc == 1) {
        ebt::usage(spec);
        exit(1);
    }

    auto args = ebt::parse_args(argc, argv, spec);

    for (int i = 0; i < argc; ++i) {
        std::cout << argv[i] << " ";
    }
    std::cout << std::endl;

    learning_env env { args };

    env.run();

    return 0;
}

learning_env::learning_env(std::unordered_map<std::string, std::string> args)
    : args(args)
{
    input_scp.open(args.at("input-scp"));
    target_scp.open(args.at("target-scp"));

    std::string line;

    std::ifstream param_ifs { args.at("param") };
    std::getline(param_ifs, line);
    layer = std::stoi(line);
    param = lstm_frame::make_tensor_tree(layer);

    tensor_tree::load_tensor(param, param_ifs);
    param_ifs.close();

    step_size = std::stod(args.at("step-size"));

    output_param = "param-last";
    if (ebt::in(std::string("output-param"), args)) {
        output_param = args.at("output-param");
    }

    output_opt_data = "opt-data-last";
    if (ebt::in(std::string("output-opt-data"), args)) {
        output_opt_data = args.at("output-opt-data");
    }

    clip = std::numeric_limits<double>::infinity();
    if (ebt::in(std::string("clip"), args)) {
        clip = std::stod(args.at("clip"));
    }

    dropout = 0;
    if (ebt::in(std::string("dropout"), args)) {
        dropout = std::stod(args.at("dropout"));
    }

    seed = 1;
    if (ebt::in(std::string("seed"), args)) {
        seed = std::stoi(args.at("seed"));
    }

    batch_size = 1;
    if (ebt::in(std::string("batch-size"), args)) {
        batch_size = std::stoi(args.at("batch-size"));
    }

    if (args.at("opt") == "rmsprop") {
        double decay = std::stod(args.at("decay"));
        opt = std::make_shared<tensor_tree::rmsprop_opt>(
            tensor_tree::rmsprop_opt(param, step_size, decay));
    } else if (args.at("opt") == "const-step") {
        opt = std::make_shared<tensor_tree::const_step_opt>(
            tensor_tree::const_step_opt(param, step_size));
    } else if (args.at("opt") == "adagrad") {
        opt = std::make_shared<tensor_tree::adagrad_opt>(
            tensor_tree::adagrad_opt(param, step_size));
    } else {
        throw std::logic_error("unknown optimizer " + args.at("opt"));
    }

    gen = std::default_random_engine { seed };

    std::ifstream opt_data_ifs { args.at("opt-data") };
    opt->load_opt_data(opt_data_ifs);
    opt_data_ifs.close();

    indices.resize(input_scp.entries.size());

    for (int i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }

    if (ebt::in(std::string("shuffle"), args)) {
        std::shuffle(indices.begin(), indices.end(), gen);
    }
}

void learning_env::run()
{
    ebt::Timer timer;

    int nsample = 0;

    std::shared_ptr<tensor_tree::vertex> accu_grad = tensor_tree::deep_copy(param);
    tensor_tree::zero(accu_grad);

    while (nsample < indices.size()) {
        std::cout << "sample: " << nsample << std::endl;
        std::cout << "index: " << indices[nsample] << std::endl;

        std::vector<std::vector<double>> input_frames = speech::load_frame_batch(input_scp.at(indices[nsample]));
        std::vector<std::vector<double>> target_frames = speech::load_frame_batch(target_scp.at(indices[nsample]));

        std::cout << "input frames: " << input_frames.size() << std::endl;
        std::cout << "target frames: " << target_frames.size() << std::endl;

        assert(input_frames.size() == target_frames.size());

        autodiff::computation_graph graph;

        std::vector<double> input_cat;
        input_cat.reserve(input_frames.size() * input_frames.front().size());

        for (int i = 0; i < input_frames.size(); ++i) {
            input_cat.insert(input_cat.end(), input_frames[i].begin(), input_frames[i].end());
        }

        std::shared_ptr<autodiff::op_t> input = graph.var(
            la::cpu::tensor<double>(la::cpu::weak_vector<double>(input_cat.data(), input_cat.size()),
            { (unsigned int) input_frames.size(), 1, (unsigned int) input_frames.front().size() }));

        input->grad_needed = false;

        std::shared_ptr<tensor_tree::vertex> var_tree = tensor_tree::make_var_tree(graph, param);

        lstm::trans_seq_t input_seq;
        input_seq.nframes = input_frames.size();
        input_seq.batch_size = batch_size;
        input_seq.dim = input_frames.front().size();
        input_seq.feat = input;
        input_seq.mask = nullptr;

        std::shared_ptr<lstm::transcriber> trans = lstm_frame::make_transcriber(
            param->children[0], dropout, &gen, false);

        lstm::trans_seq_t feat_seq = (*trans)(var_tree->children[0], input_seq);

        lstm::fc_transcriber fc_trans { (int) input_frames.front().size() };
        auto output_seq = fc_trans(var_tree->children[1], feat_seq);

        std::shared_ptr<autodiff::op_t> output = output_seq.feat;
        auto& pred = autodiff::get_output<la::cpu::tensor_like<double>>(output);

        std::vector<double> gold_vec;
        gold_vec.reserve(target_frames.size() * target_frames.front().size());

        for (int i = 0; i < target_frames.size(); ++i) {
            gold_vec.insert(gold_vec.end(), target_frames[i].begin(), target_frames[i].end());
        }

        la::cpu::weak_tensor<double> gold { gold_vec.data(),
            {(unsigned int) target_frames.size(), 1, (unsigned int) target_frames.front().size()}};

        std::cout << "first frame pred: ";
        for (int j = 0; j < 10; ++j) {
            std::cout << pred({0, 0, j}) << " ";
        }
        std::cout << std::endl;

        std::cout << "first frame gold: ";
        for (int j = 0; j < 10; ++j) {
            std::cout << gold({0, 0, j}) << " ";
        }
        std::cout << std::endl;

        nn::l2_loss loss { gold, pred };

        output->grad = std::make_shared<la::cpu::tensor<double>>(loss.grad());
        
        double ell = loss.loss();

        if (std::isnan(ell)) {
            throw std::logic_error("loss is nan");
        }

        std::cout << "loss: " << ell / batch_size << std::endl;
        std::cout << "E: " << ell / target_frames.size() << std::endl;

        auto topo_order = autodiff::natural_topo_order(graph);

        autodiff::guarded_grad(topo_order, autodiff::grad_funcs);

        std::shared_ptr<tensor_tree::vertex> grad = lstm_frame::make_tensor_tree(layer);

        tensor_tree::copy_grad(grad, var_tree);

        tensor_tree::axpy(accu_grad, 1, grad);

        if (nsample % batch_size == batch_size - 1) {

            tensor_tree::axpy(accu_grad, 1.0 / batch_size - 1, accu_grad);

            double n = tensor_tree::norm(accu_grad);

            std::cout << "grad norm: " << n << std::endl;

            if (ebt::in(std::string("clip"), args)) {
                if (n > clip) {
                    tensor_tree::axpy(accu_grad, clip / n - 1, accu_grad);
                    std::cout << "gradient clipped" << std::endl;
                }
            }

            std::vector<std::shared_ptr<tensor_tree::vertex>> vars = tensor_tree::leaves_pre_order(param);

            la::cpu::tensor<double> const& v = tensor_tree::get_tensor(vars.front());

            double v1 = v.data()[0];

            opt->update(accu_grad);

            tensor_tree::zero(accu_grad);

            double v2 = v.data()[0];

            std::cout << "name: " << vars.front()->name << " weight: " << v1
                << " update: " << v2 - v1 << " rate: " << (v2 - v1) / v1 << std::endl;

            std::cout << std::endl;

        }

        ++nsample;

#if DEBUG_TOP
        if (nsample >= DEBUG_TOP) {
            break;
        }
#endif
    }

    std::ofstream param_ofs { output_param };
    param_ofs << layer << std::endl;
    tensor_tree::save_tensor(param, param_ofs);
    param_ofs.close();

    std::ofstream opt_data_ofs { output_opt_data };
    opt->save_opt_data(opt_data_ofs);
    opt_data_ofs.close();
}

