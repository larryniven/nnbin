#include "la/la.h"
#include "autodiff/autodiff.h"
#include "ebt/ebt.h"
#include "speech/speech.h"
#include <fstream>
#include <vector>
#include "nn/nn.h"
#include <random>
#include "nn/tensor-tree.h"
#include "nn/cnn.h"
#include "nn/cnn-frame.h"
#include <algorithm>

struct learning_env {

    speech::batch_indices frame_batch;
    speech::batch_indices label_batch;

    std::shared_ptr<tensor_tree::vertex> param;

    std::shared_ptr<tensor_tree::optimizer> opt;

    std::shared_ptr<tensor_tree::vertex> var_tree;

    cnn::cnn_t cnn_config;

    double step_size;
    double decay;

    double dropout;
    int seed;

    std::string output_param;
    std::string output_opt_data;

    double clip;

    std::default_random_engine gen;

    std::unordered_map<std::string, int> label_id;

    std::unordered_set<std::string> ignored;

    std::unordered_map<std::string, std::string> args;

    learning_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "frame-cnn-learn",
        "Train a CNN frame classifier",
        {
            {"frame-batch", "", true},
            {"label-batch", "", true},
            {"param", "", true},
            {"opt-data", "", true},
            {"output-param", "", false},
            {"output-opt-data", "", false},
            {"label", "", true},
            {"ignore", "", false},
            {"dropout", "", false},
            {"seed", "", false},
            {"shuffle", "", false},
            {"opt", "const-step,adagrad,rmsprop,adam", true},
            {"step-size", "", true},
            {"momentum", "", false},
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
    frame_batch.open(args.at("frame-batch"));
    label_batch.open(args.at("label-batch"));

    std::ifstream param_ifs { args.at("param") };
    cnn_config = cnn::load_param(param_ifs);
    param = cnn_config.param;
    param_ifs.close();

    step_size = std::stod(args.at("step-size"));

    if (ebt::in(std::string("decay"), args)) {
        decay = std::stod(args.at("decay"));
    }

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

    std::vector<std::string> label_vec = speech::load_label_set(args.at("label"));
    for (int i = 0; i < label_vec.size(); ++i) {
        label_id[label_vec[i]] = i;
    }

    if (ebt::in(std::string("ignore"), args)) {
        auto parts = ebt::split(args.at("ignore"), ",");
        ignored.insert(parts.begin(), parts.end());
    }

    dropout = 0;
    if (ebt::in(std::string("dropout"), args)) {
        dropout = std::stod(args.at("dropout"));
    }

    seed = 1;
    if (ebt::in(std::string("seed"), args)) {
        seed = std::stoi(args.at("seed"));
    }

    if (args.at("opt") == "rmsprop") {
        double decay = std::stod(args.at("decay"));
        opt = std::make_shared<tensor_tree::rmsprop_opt>(
            tensor_tree::rmsprop_opt(param, step_size, decay));
    } else if (args.at("opt") == "const-step") {
        opt = std::make_shared<tensor_tree::const_step_opt>(
            tensor_tree::const_step_opt(param, step_size));
    } else if (args.at("opt") == "const-step-momentum") {
        double momentum = std::stod(args.at("momentum"));
        opt = std::make_shared<tensor_tree::const_step_momentum_opt>(
            tensor_tree::const_step_momentum_opt(param, step_size, momentum));
    } else if (args.at("opt") == "adagrad") {
        opt = std::make_shared<tensor_tree::adagrad_opt>(
            tensor_tree::adagrad_opt(param, step_size));
    } else {
        std::cout << "unknown optimizer " << args.at("opt") << std::endl;
        exit(1);
    }

    std::ifstream opt_data_ifs { args.at("opt-data") };
    opt->load_opt_data(opt_data_ifs);
    opt_data_ifs.close();

    gen = std::default_random_engine { seed };

    if (ebt::in(std::string("shuffle"), args)) {
        std::vector<int> indices;
        indices.resize(frame_batch.pos.size());

        for (int i = 0; i < indices.size(); ++i) {
            indices[i] = i;
        }
        std::shuffle(indices.begin(), indices.end(), gen);

        std::vector<unsigned long> pos = frame_batch.pos;
        for (int i = 0; i < indices.size(); ++i) {
            frame_batch.pos[i] = pos[indices[i]];
        }

        pos = label_batch.pos;
        for (int i = 0; i < indices.size(); ++i) {
            label_batch.pos[i] = pos[indices[i]];
        }
    }
}

void learning_env::run()
{
    ebt::Timer timer;

    int nsample = 0;

    while (nsample < frame_batch.pos.size()) {
        std::vector<std::vector<double>> frames;

        frames = speech::load_frame_batch(frame_batch.at(nsample));

	std::vector<std::string> labels = speech::load_label_batch(label_batch.at(nsample));

        std::cout << "sample: " << nsample << std::endl;

        assert(frames.size() == labels.size());

        autodiff::computation_graph graph;
        var_tree = tensor_tree::make_var_tree(graph, param);

        la::cpu::tensor<double> input_tensor;
        input_tensor.resize({ (unsigned int) frames.size(), 40, 3});

        for (int t = 0; t < frames.size(); ++t) {
            for (int i = 0; i < frames.front().size(); ++i) {
                input_tensor({t, i % 40, i / 40}) = frames[t][i];
            }
        }

        std::shared_ptr<autodiff::op_t> input = graph.var(input_tensor);

        std::shared_ptr<cnn::transcriber> trans = cnn::make_transcriber(cnn_config, dropout, &gen);
        std::shared_ptr<autodiff::op_t> hidden = (*trans)(var_tree, input);

        auto& t = autodiff::get_output<la::cpu::tensor_like<double>>(hidden);

        std::cout << std::endl;
        for (int i = 0; i < 20; ++i) {
            for (int j = 0; j < 40; ++j) {
                std::cout << t({i, j}) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;

        trans = std::make_shared<cnn::logsoftmax_transcriber>(
            cnn::logsoftmax_transcriber{});

        std::shared_ptr<autodiff::op_t> logprob = (*trans)(var_tree->children.back(), hidden);

        std::vector<double> gold_vec;
        gold_vec.resize(labels.size() * label_id.size());
        int nframes = 0;

        for (int t = 0; t < labels.size(); ++t) {
            if (!ebt::in(labels[t], ignored)) {
                gold_vec[t * label_id.size() + label_id.at(labels[t])] = 1;
                nframes += 1;
            }
        }

        la::cpu::weak_tensor<double> gold { gold_vec.data(),
            {(unsigned int) labels.size(), (unsigned int) label_id.size()}};
        auto& pred = autodiff::get_output<la::cpu::tensor_like<double>>(logprob);
        nn::log_loss loss { gold, pred };

        logprob->grad = std::make_shared<la::cpu::tensor<double>>(loss.grad());
        
        double ell = loss.loss();

        if (std::isnan(ell)) {
            std::cerr << "loss is nan" << std::endl;
            exit(1);
        }

        std::cout << "loss: " << ell << std::endl;
        std::cout << "E: " << ell / nframes << std::endl;

        auto topo_order = autodiff::natural_topo_order(graph);
        autodiff::guarded_grad(topo_order, autodiff::grad_funcs);

        std::shared_ptr<tensor_tree::vertex> grad = cnn::make_tensor_tree(cnn_config.conv_layer, cnn_config.fc_layer);
        tensor_tree::copy_grad(grad, var_tree);

        double n = tensor_tree::norm(grad);

        std::cout << "grad norm: " << n << std::endl;

        if (ebt::in(std::string("clip"), args)) {
            if (n > clip) {
                tensor_tree::imul(grad, clip / n);
                std::cout << "gradient clipped" << std::endl;
            }
        }

        std::vector<std::shared_ptr<tensor_tree::vertex>> vars = tensor_tree::leaves_pre_order(param);
        la::cpu::tensor<double> const& v = tensor_tree::get_tensor(vars.front());

        double v1 = v.data()[0];

        opt->update(grad);

        double v2 = v.data()[0];

        std::cout << vars.front()->name << " weight: " << v1 << " update: " << v2 - v1 << " rate: " << (v2 - v1) / v1 << std::endl;

        std::cout << std::endl;

#if DEBUG_TOP
        if (nsample == DEBUG_TOP) {
            break;
        }
#endif

        ++nsample;
    }

    std::ofstream param_ofs { output_param };
    cnn::save_param(cnn_config, param_ofs);
    param_ofs.close();

    std::ofstream opt_data_ofs { output_opt_data };
    opt->save_opt_data(opt_data_ofs);
    opt_data_ofs.close();
}

