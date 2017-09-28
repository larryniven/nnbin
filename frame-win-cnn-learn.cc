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

    unsigned int input_channel;

    std::shared_ptr<tensor_tree::vertex> param;

    std::shared_ptr<tensor_tree::optimizer> opt;

    cnn::cnn_t cnn_config;

    double step_size;
    double decay;

    unsigned int win_size;

    double dropout;
    int seed;

    std::string output_param;
    std::string output_opt_data;

    double clip;

    unsigned int batch_size;

    std::default_random_engine gen;

    std::vector<std::pair<int, int>> indices;

    std::unordered_map<std::string, int> label_id;

    std::unordered_set<std::string> ignored;

    std::unordered_map<std::string, std::string> args;

    learning_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "frame-win-cnn-learn",
        "Train a CNN frame classifier",
        {
            {"frame-batch", "", true},
            {"label-batch", "", true},
            {"input-channel", "", true},
            {"param", "", true},
            {"opt-data", "", true},
            {"output-param", "", false},
            {"output-opt-data", "", false},
            {"win-size", "", true},
            {"label", "", true},
            {"ignore", "", false},
            {"dropout", "", false},
            {"batch-size", "", false},
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

    input_channel = std::stoi(args.at("input-channel"));

    std::ifstream param_ifs { args.at("param") };
    cnn_config = cnn::load_param(param_ifs);
    param = cnn_config.param;
    param_ifs.close();

    win_size = std::stoi(args.at("win-size"));

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

    batch_size = 1;
    if (ebt::in(std::string("batch-size"), args)) {
        batch_size = std::stoi(args.at("batch-size"));
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

    for (int i = 0; i < label_batch.pos.size(); ++i) {
        std::vector<std::string> labels = speech::load_label_batch(label_batch.at(i));

        for (int t = 0; t < labels.size(); ++t) {
            indices.push_back(std::make_pair(i, t));
        }
    }

    if (ebt::in(std::string("shuffle"), args)) {
        std::shuffle(indices.begin(), indices.end(), gen);
    }
}

void learning_env::run()
{
    ebt::Timer timer;

    int nsample = 0;

    while (nsample < indices.size()) {
        autodiff::computation_graph graph;
        std::shared_ptr<tensor_tree::vertex> var_tree = tensor_tree::make_var_tree(graph, param);

        std::vector<double> input_tensor_vec;
        std::vector<double> gold_vec;

        unsigned int input_dim = 0;
        unsigned int loaded_sample = 0;
        while (nsample < indices.size() && loaded_sample < batch_size) {
            std::vector<std::vector<double>> frames = speech::load_frame_batch(
                frame_batch.at(indices.at(nsample).first));

	    std::vector<std::string> labels = speech::load_label_batch(
                label_batch.at(indices.at(nsample).first));

            assert(frames.size() == labels.size());

            input_dim = frames.front().size() / input_channel;

            int t = indices.at(nsample).second;

            ++nsample;

            if (ebt::in(labels[t], ignored)) {
                continue;
            }

            for (int i = 0; i < win_size; ++i) {
                if (0 <= t + i - win_size / 2 && t + i - win_size / 2 < frames.size()) {
                    for (int j = 0; j < frames.front().size(); ++j) {
                        input_tensor_vec.push_back(frames[t + i - win_size / 2][j]);
                    }
                } else {
                    for (int j = 0; j < frames.front().size(); ++j) {
                        input_tensor_vec.push_back(0);
                    }
                }
            }

            for (int i = 0; i < label_id.size(); ++i) {
                gold_vec.push_back((i == label_id.at(labels[t]) ? 1 : 0));
            }

            ++loaded_sample;
        }

        std::cout << "loaded sample: " << loaded_sample << std::endl;

        la::cpu::tensor<double> input_tensor { la::cpu::vector<double>(input_tensor_vec),
            std::vector<unsigned int> { loaded_sample, win_size, input_dim, input_channel }};

        la::cpu::tensor<double> gold { la::cpu::vector<double>(gold_vec),
            std::vector<unsigned int> { loaded_sample, (unsigned int) label_id.size() }};

        std::shared_ptr<autodiff::op_t> input = graph.var(input_tensor);

        std::shared_ptr<cnn::transcriber> trans = cnn::make_transcriber(cnn_config, dropout, &gen);
        std::shared_ptr<autodiff::op_t> logprob = (*trans)(var_tree, input);

        auto& pred = autodiff::get_output<la::cpu::tensor_like<double>>(logprob);
        nn::log_loss loss { gold, pred };

        logprob->grad = std::make_shared<la::cpu::tensor<double>>(loss.grad());
        
        double ell = loss.loss();

        if (std::isnan(ell)) {
            std::cerr << "loss is nan" << std::endl;
            exit(1);
        }

        std::cout << "loss: " << ell / loaded_sample << std::endl;

        auto topo_order = autodiff::natural_topo_order(graph);
        autodiff::guarded_grad(topo_order, autodiff::grad_funcs);

        std::shared_ptr<tensor_tree::vertex> grad = cnn::make_tensor_tree(cnn_config);
        tensor_tree::copy_grad(grad, var_tree);

#if 0
        {
            std::shared_ptr<tensor_tree::vertex> param2 = tensor_tree::deep_copy(param);
            la::cpu::tensor<double>& t = tensor_tree::get_tensor(param2->children[0]->children[0]);
            t({0, 0, 0, 0}) += 1e-8;

            autodiff::computation_graph graph2;
            auto var_tree2 = tensor_tree::make_var_tree(graph2, param2);
            std::shared_ptr<autodiff::op_t> input2 = graph2.var(input_tensor);

            std::shared_ptr<cnn::transcriber> trans2 = cnn::make_transcriber(cnn_config, dropout, &gen);
            std::shared_ptr<autodiff::op_t> logprob2 = (*trans)(var_tree2, input2);

            auto& pred2 = autodiff::get_output<la::cpu::tensor_like<double>>(logprob2);
            nn::log_loss loss2 { gold, pred2 };

            double ell2 = loss2.loss();

            std::cout << "numeric grad: " << (ell2 - ell) / 1e-8 << std::endl;

            la::cpu::tensor<double>& grad_t = tensor_tree::get_tensor(grad->children[0]->children[0]);
            std::cout << "analytic grad: " << grad_t({0, 0, 0, 0}) << std::endl;
        }
#endif

        tensor_tree::imul(grad, 1.0 / loaded_sample);

        double n = tensor_tree::norm(grad);

        std::cout << "grad norm: " << n << std::endl;

        if (ebt::in(std::string("clip"), args)) {
            if (n > clip) {
                tensor_tree::imul(grad, clip / n);
                std::cout << "gradient clipped" << std::endl;
            }
        }

        std::vector<std::shared_ptr<tensor_tree::vertex>> vars
            = tensor_tree::leaves_pre_order(param);

        la::cpu::tensor<double> const& v = tensor_tree::get_tensor(vars.front());

        double v1 = v.data()[0];

        opt->update(grad);

        double v2 = v.data()[0];

        std::cout << vars.front()->name << " weight: " << v1
            << " update: " << v2 - v1 << " rate: " << (v2 - v1) / v1 << std::endl;

        std::cout << std::endl;
    }

    std::ofstream param_ofs { output_param };
    cnn::save_param(cnn_config, param_ofs);
    param_ofs.close();

    std::ofstream opt_data_ofs { output_opt_data };
    opt->save_opt_data(opt_data_ofs);
    opt_data_ofs.close();
}

