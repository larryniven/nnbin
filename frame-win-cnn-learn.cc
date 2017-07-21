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
#include <iomanip>

struct learning_env {

    speech::batch_indices frame_batch;
    speech::batch_indices label_batch;

    std::vector<std::pair<int, int>> indices;
    int win_size;
    unsigned int input_channel;

    std::shared_ptr<tensor_tree::vertex> param;

    std::shared_ptr<tensor_tree::optimizer> opt;

    std::shared_ptr<tensor_tree::vertex> var_tree;

    cnn::cnn_t cnn_config;

    double step_size;
    double decay;

    double dropout;
    int seed;

    std::string output_param_prefix;
    std::string output_opt_data_prefix;

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
        "frame-win-cnn-learn",
        "Train a CNN frame classifier",
        {
            {"frame-batch", "", true},
            {"label-batch", "", true},
            {"win-size", "", true},
            {"input-channel", "", true},
            {"param", "", true},
            {"opt-data", "", true},
            {"output-param-prefix", "", false},
            {"output-opt-data-prefix", "", false},
            {"label", "", true},
            {"ignore", "", false},
            {"dropout", "", false},
            {"seed", "", false},
            {"rand-state", "", false},
            {"shuffle", "", false},
            {"save-every", "", false},
            {"start-from", "", false},
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

    win_size = std::stoi(args.at("win-size"));
    input_channel = std::stoi(args.at("input-channel"));

    std::ifstream param_ifs { args.at("param") };
    cnn_config = cnn::load_param(param_ifs);
    param = cnn_config.param;
    param_ifs.close();

    step_size = std::stod(args.at("step-size"));

    if (ebt::in(std::string("decay"), args)) {
        decay = std::stod(args.at("decay"));
    }

    output_param_prefix = "param-";
    if (ebt::in(std::string("output-param-prefix"), args)) {
        output_param_prefix = args.at("output-param-prefix");
    }

    output_opt_data_prefix = "opt-data-";
    if (ebt::in(std::string("output-opt-data-prefix"), args)) {
        output_opt_data_prefix = args.at("output-opt-data-prefix");
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
        std::vector<int> nframes;

        for (int i = 0; i < label_batch.pos.size(); ++i) {
            std::vector<std::string> labels = speech::load_label_batch(label_batch.at(i));
            nframes.push_back(labels.size());
        }

        for (int i = 0; i < label_batch.pos.size(); ++i) {
            for (int j = 0; j < nframes[i]; ++j) {
                indices.push_back(std::make_pair(i, j));
            }
        }

        std::shuffle(indices.begin(), indices.end(), gen);
    }
}

void learning_env::run()
{
    std::time_t now = std::time(nullptr);
    std::cout << "start time: " << std::put_time(std::localtime(&now), "%c %Z") << std::endl;
    std::cout << std::endl;

    ebt::Timer timer;

    int nsample = 0;

    if (ebt::in(std::string("start-from"), args)) {
        nsample = std::stoi(args.at("start-from"));
    }

    if (ebt::in(std::string("rand-state"), args)) {
        std::istringstream iss { args.at("rand-state") };
        iss >> gen;
    }

    while (nsample < indices.size()) {

        std::vector<std::vector<double>> frames = speech::load_frame_batch(frame_batch.at(indices[nsample].first));

	std::vector<std::string> labels = speech::load_label_batch(label_batch.at(indices[nsample].first));

        int frame_index = indices[nsample].second;

        std::cout << "sample: " << nsample
            << " utt: " << indices[nsample].first
            << " frame: " << indices[nsample].second
            << " label: " << labels[frame_index] << std::endl;

        assert(frames.size() == labels.size());

        if (ebt::in(labels[frame_index], ignored)) {
            std::cout << "skip " << labels[frame_index] << std::endl;
            std::cout << std::endl;
            ++nsample;
            continue;
        }

        autodiff::computation_graph graph;
        var_tree = tensor_tree::make_var_tree(graph, param);

        assert(frames.front().size() % input_channel == 0);

        unsigned int input_dim = frames.front().size() / input_channel;

        la::cpu::tensor<double> input_tensor;
        input_tensor.resize({ (unsigned int) win_size, input_dim, input_channel});

        for (int t = frame_index - win_size / 2; t <= frame_index + win_size / 2; ++t) {
            for (int i = 0; i < frames.front().size(); ++i) {
                if (t < 0 || t >= frames.size()) {
                    input_tensor({t - frame_index + win_size / 2, int(i % input_dim), int(i / input_dim)}) = 0;
                } else {
                    input_tensor({t - frame_index + win_size / 2, int(i % input_dim), int(i / input_dim)}) = frames[t][i];
                }
            }
        }

        std::shared_ptr<autodiff::op_t> input = graph.var(input_tensor);

        std::shared_ptr<cnn::transcriber> trans = cnn::make_transcriber(cnn_config, dropout, &gen);
        std::shared_ptr<autodiff::op_t> logprob = (*trans)(var_tree, input);

/*
        auto& t = autodiff::get_output<la::cpu::tensor_like<double>>(logprob);

        std::cout << std::endl;
        for (int i = 0; i < 20; ++i) {
            for (int j = 0; j < 40; ++j) {
                std::cout << t({i, j}) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
*/

        std::vector<double> gold_vec;
        gold_vec.resize(label_id.size());
        gold_vec[label_id.at(labels[frame_index])] = 1;

        la::cpu::weak_tensor<double> gold { gold_vec.data(),
            {(unsigned int) label_id.size()}};
        auto& pred = autodiff::get_output<la::cpu::tensor_like<double>>(logprob);
        nn::log_loss loss { gold, pred };

        logprob->grad = std::make_shared<la::cpu::tensor<double>>(loss.grad());
        
        double ell = loss.loss();

        if (std::isnan(ell)) {
            throw std::logic_error("loss is nan");
        }

        std::cout << "loss: " << ell << std::endl;

        auto topo_order = autodiff::natural_topo_order(graph);
        autodiff::guarded_grad(topo_order, autodiff::grad_funcs);

        std::shared_ptr<tensor_tree::vertex> grad = cnn::make_tensor_tree(cnn_config);
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

        std::cout << "param norm: " << tensor_tree::norm(param) << std::endl;

#if DEBUG_TOP
        if (nsample == DEBUG_TOP) {
            break;
        }
#endif

        ++nsample;

        std::cout << "rand: " << gen << std::endl;

        if (ebt::in(std::string("save-every"), args) && nsample % std::stoi(args.at("save-every")) == 0) {
            std::ofstream param_ofs { output_param_prefix + std::to_string(nsample) };
            cnn::save_param(cnn_config, param_ofs);
            param_ofs.close();

            std::ofstream opt_data_ofs { output_opt_data_prefix + std::to_string(nsample) };
            opt->save_opt_data(opt_data_ofs);
            opt_data_ofs.close();
        }

        std::cout << std::endl;
    }

    std::ofstream param_ofs { output_param_prefix + "final" };
    cnn::save_param(cnn_config, param_ofs);
    param_ofs.close();

    std::ofstream opt_data_ofs { output_opt_data_prefix + "final" };
    opt->save_opt_data(opt_data_ofs);
    opt_data_ofs.close();
}

