#include "la/la-cpu.h"
#include "autodiff/autodiff.h"
#include "ebt/ebt.h"
#include "speech/speech.h"
#include <fstream>
#include <vector>
#include "nn/lstm.h"
#include "nn/nn.h"
#include <random>
#include "nn/tensor-tree.h"
#include "nn/lstm-frame.h"
#include <algorithm>

using namespace std::literals::string_literals;

struct learning_env {

    speech::batch_indices frame_batch;
    speech::batch_indices label_batch;

    std::vector<int> indices;

    int layer;
    std::shared_ptr<tensor_tree::vertex> param;

    std::shared_ptr<tensor_tree::optimizer> opt;

    std::string output_param;
    std::string output_opt_data;

    int batch_size;

    int seed;
    double dropout;
    double step_size;
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
        "learn-lstm",
        "Train a LSTM frame classifier",
        {
            {"frame-batch", "", true},
            {"label-batch", "", true},
            {"param", "", true},
            {"opt-data", "", true},
            {"output-param", "", false},
            {"output-opt-data", "", false},
            {"label", "", true},
            {"ignore", "", false},
            {"batch-size", "", false},
            {"dropout", "", false},
            {"shuffle", "", false},
            {"seed", "", false},
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
    frame_batch.open(args.at("frame-batch"));
    label_batch.open(args.at("label-batch"));

    std::string line;

    std::ifstream param_ifs { args.at("param") };
    std::getline(param_ifs, line);
    layer = std::stoi(line);
    param = lstm_frame::make_tensor_tree(layer);
    tensor_tree::load_tensor(param, param_ifs);
    param_ifs.close();

    step_size = std::stod(args.at("step-size"));

    output_param = "param-last";
    if (ebt::in("output-param"s, args)) {
        output_param = args.at("output-param");
    }

    output_opt_data = "opt-data-last";
    if (ebt::in("output-opt-data"s, args)) {
        output_opt_data = args.at("output-opt-data");
    }

    clip = std::numeric_limits<double>::infinity();
    if (ebt::in("clip"s, args)) {
        clip = std::stod(args.at("clip"));
    }

    std::vector<std::string> label_vec = speech::load_label_set(args.at("label"));
    for (int i = 0; i < label_vec.size(); ++i) {
        label_id[label_vec[i]] = i;
    }

    if (ebt::in("ignore"s, args)) {
        auto parts = ebt::split(args.at("ignore"), ",");
        ignored.insert(parts.begin(), parts.end());
    }

    batch_size = 1;
    if (ebt::in("batch-size"s, args)) {
        batch_size = std::stoi(args.at("batch-size"));
    }

    dropout = 0;
    if (ebt::in("dropout"s, args)) {
        dropout = std::stod(args.at("dropout"));
    }

    seed = 1;
    if (ebt::in("seed"s, args)) {
        seed = std::stoi(args.at("seed"));
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
        std::cout << "unknown optimizer " << args.at("opt") << std::endl;
        exit(1);
    }

    gen = std::default_random_engine { seed };

    std::ifstream opt_data_ifs { args.at("opt-data") };
    std::getline(opt_data_ifs, line);
    opt->load_opt_data(opt_data_ifs);
    opt_data_ifs.close();

    indices.resize(frame_batch.pos.size());
    for (int i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }

    if (ebt::in(std::string("shuffle"), args)) {
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
        std::vector<std::vector<std::vector<double>>> frames;
	std::vector<std::vector<std::string>> labels;

        for (int i = nsample; i < std::min<int>(nsample + batch_size, frame_batch.pos.size()); ++i) {
            frames.push_back(speech::load_frame_batch(frame_batch.at(i)));
            labels.push_back(speech::load_label_batch(label_batch.at(i)));
        }

        assert(frames.size() == labels.size());

        unsigned int batch_size_local = frames.size();

        std::cout << "sample: ";
        for (int i = 0; i < batch_size_local; ++i) {
            std::cout << indices[nsample + i] << " ";
        }
        std::cout << std::endl;
        std::cout << "rand state: "<< gen << std::endl;

#if 0
        std::ofstream param_ofs { "param-debug" };
        param_ofs << layer << std::endl;
        tensor_tree::save_tensor(param, param_ofs);
        param_ofs.close();

        std::ofstream opt_data_ofs { "opt-data-debug" };
        opt_data_ofs << layer << std::endl;
        opt->save_opt_data(opt_data_ofs);
        opt_data_ofs.close();
#endif

        autodiff::computation_graph graph;

        unsigned int max_len = 0;

        std::cout << "frames: ";
        for (int b = 0; b < batch_size_local; ++b) {
            if (frames[b].size() > max_len) {
                max_len = frames[b].size();
            }

            std::cout << frames[b].size() << " ";
        }
        std::cout << std::endl;

        unsigned int ndim = frames.front().front().size();

        std::vector<double> input_vec;
        input_vec.reserve(max_len * batch_size_local * ndim);

        std::vector<double> zeros;
        zeros.resize(ndim);

        for (int t = 0; t < max_len; ++t) {
            for (int b = 0; b < batch_size_local; ++b) {
                if (t < frames[b].size()) {
                    assert(frames[b][t].size() == ndim);
                    input_vec.insert(input_vec.end(), frames[b][t].begin(), frames[b][t].end());
                } else {
                    input_vec.insert(input_vec.end(), zeros.begin(), zeros.end());
                }
            }
        }

        std::shared_ptr<autodiff::op_t> input = graph.var(la::cpu::tensor<double>(
            la::cpu::weak_tensor<double>(input_vec.data(), {max_len, batch_size_local, ndim})));

        input->grad_needed = false;

        std::vector<double> mask_vec;
        mask_vec.reserve(max_len * batch_size_local);

        for (int t = 0; t < max_len; ++t) {
            for (int b = 0; b < batch_size_local; ++b) {
                if (t < frames[b].size()) {
                    mask_vec.push_back(1);
                } else {
                    mask_vec.push_back(0);
                }
            }
        }

        std::shared_ptr<autodiff::op_t> mask = graph.var(la::cpu::tensor<double>(
            la::cpu::weak_tensor<double>(mask_vec.data(), {max_len, batch_size_local})));

        mask->grad_needed = false;

        std::shared_ptr<tensor_tree::vertex> var_tree = tensor_tree::make_var_tree(graph, param);

        std::shared_ptr<lstm::transcriber> trans
            = lstm_frame::make_transcriber(layer, dropout, &gen);

        std::shared_ptr<autodiff::op_t> hidden;
        std::shared_ptr<autodiff::op_t> ignore;

        std::tie(hidden, ignore) = (*trans)(var_tree->children[0], input, mask);

        trans = std::make_shared<lstm::logsoftmax_transcriber>(
            lstm::logsoftmax_transcriber { nullptr });

        std::shared_ptr<autodiff::op_t> logprob;

        std::tie(logprob, ignore) = (*trans)(var_tree, hidden, mask);

        int nframes = 0;

        la::cpu::tensor<double> gold;
        gold.resize({max_len, batch_size_local, (unsigned int) label_id.size()});

        for (int t = 0; t < max_len; ++t) {
            for (int b = 0; b < batch_size_local; ++b) {
                if (t < labels[b].size() && !ebt::in(labels[b][t], ignored)) {
                    gold({t, b, label_id.at(labels[b][t])}) = 1;
                    nframes += 1;
                }
            }
        }

        auto& pred = autodiff::get_output<la::cpu::tensor_like<double>>(logprob);

        nn::log_loss loss { gold, pred };
        logprob->grad = std::make_shared<la::cpu::tensor<double>>(loss.grad());

        double ell = loss.loss();

        if (std::isnan(ell)) {
            std::cerr << "loss is nan" << std::endl;
            exit(1);
        }

        std::cout << "loss: " << ell / batch_size_local << std::endl;
        std::cout << "E: " << ell / nframes << std::endl;

        auto topo_order = autodiff::natural_topo_order(graph);
        autodiff::guarded_grad(topo_order, autodiff::grad_funcs);

#if 0
        for (int i = 0; i < 3; ++i) {
            auto& t = autodiff::get_grad<la::tensor_like<double>>(logprob[logprob.size() - i - 1]);

            for (int j = 0; j < t.vec_size(); ++j) {
                std::cout << t.data()[j] << " ";
            }
            std::cout << std::endl;
        }
#endif

        std::shared_ptr<tensor_tree::vertex> grad = lstm_frame::make_tensor_tree(layer);
        tensor_tree::copy_grad(grad, var_tree);

        tensor_tree::imul(grad, 1.0 / batch_size_local);

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

        std::cout << "weight: " << v1 << " update: " << v2 - v1 << " rate: " << (v2 - v1) / v1 << std::endl;

        std::cout << std::endl;

        nsample += batch_size_local;

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
    opt_data_ofs << layer << std::endl;
    opt->save_opt_data(opt_data_ofs);
    opt_data_ofs.close();
}

