#include "la/la.h"
#include "autodiff/autodiff.h"
#include "ebt/ebt.h"
#include "speech/speech.h"
#include <fstream>
#include <vector>
#include "nn/lstm.h"
#include "nn/nn.h"
#include <random>
#include "nn/tensor-tree.h"
#include "nn/cnn.h"
#include <algorithm>

std::shared_ptr<tensor_tree::vertex> make_tensor_tree(int conv_layer, int fc_layer)
{
    tensor_tree::vertex root { tensor_tree::tensor_t::nil };

    root.children.push_back(cnn::make_cnn_tensor_tree(conv_layer));

    for (int i = 0; i < fc_layer; ++i) {
        tensor_tree::vertex fc { tensor_tree::tensor_t::nil };
        fc.children.push_back(tensor_tree::make_tensor("softmax weight"));
        fc.children.push_back(tensor_tree::make_tensor("softmax bias"));
        root.children.push_back(std::make_shared<tensor_tree::vertex>(fc));
    }

    return std::make_shared<tensor_tree::vertex>(root);
}

struct learning_env {

    speech::batch_indices frame_batch;
    speech::batch_indices label_batch;

    std::shared_ptr<tensor_tree::vertex> param;

    std::shared_ptr<tensor_tree::optimizer> opt;

    std::shared_ptr<tensor_tree::vertex> var_tree;

    int conv_layer;
    int fc_layer;

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
            {"step-size", "", true},
            {"decay", "", false},
            {"output-param", "", false},
            {"output-opt-data", "", false},
            {"clip", "", false},
            {"label", "", true},
            {"ignore", "", false},
            {"dropout", "", false},
            {"seed", "", false},
            {"shuffle", "", false},
            {"const-step-update"},
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
    std::vector<std::string> parts = ebt::split(line);
    conv_layer = std::stoi(parts[0]);
    fc_layer = std::stoi(parts[1]);
    param = make_tensor_tree(conv_layer, fc_layer);
    tensor_tree::load_tensor(param, param_ifs);
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

    if (ebt::in(std::string("dropout"), args)) {
        dropout = std::stod(args.at("dropout"));
    }

    if (ebt::in(std::string("seed"), args)) {
        seed = std::stoi(args.at("seed"));
    }

    if (ebt::in(std::string("decay"), args)) {
        opt = std::make_shared<tensor_tree::rmsprop_opt>(
            tensor_tree::rmsprop_opt(param, decay, step_size));
    } else if (ebt::in(std::string("const-step-update"), args)) {
        opt = std::make_shared<tensor_tree::const_step_opt>(
            tensor_tree::const_step_opt(param, step_size));
    } else {
        opt = std::make_shared<tensor_tree::adagrad_opt>(
            tensor_tree::adagrad_opt(param, step_size));
    }

    std::ifstream opt_data_ifs { args.at("opt-data") };
    std::getline(opt_data_ifs, line);
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

        la::tensor<double> input_tensor;
        input_tensor.resize({ (unsigned int) frames.size(), (unsigned int) frames.front().size(), 1});

        for (int t = 0; t < frames.size(); ++t) {
            for (int i = 0; i < frames.front().size(); ++i) {
                input_tensor({t, i, 0}) = frames[t][i];
            }
        }

        std::shared_ptr<autodiff::op_t> input = graph.var(input_tensor);

        cnn::multilayer_transcriber trans;

        for (int i = 0; i < conv_layer; ++i) {
            trans.layers.push_back(std::make_shared<cnn::cnn_transcriber>(
                cnn::cnn_transcriber{}));
        }

        std::shared_ptr<autodiff::op_t> feat = trans(input, var_tree->children[0]);

        auto& t = tensor_tree::get_tensor(param->children[0]->children.back()->children.back());

        feat = autodiff::reshape(feat,
            { (unsigned int) frames.size(), (unsigned int) frames.front().size() * t.size(0) });

        std::vector<std::shared_ptr<autodiff::op_t>> logprob;

        for (int t = 0; t < frames.size(); ++t) {
            std::shared_ptr<autodiff::op_t> feat_t = autodiff::row_at(feat, t);

            for (int i = 0; i < fc_layer - 1; ++i) {
                if (ebt::in(std::string("dropout"), args)) {
                    feat_t = autodiff::emul(feat_t, autodiff::dropout_mask(feat_t, dropout, gen));
                }

                feat_t = autodiff::relu(autodiff::add(
                    autodiff::mul(feat_t, tensor_tree::get_var(var_tree->children[i + 1]->children[0])),
                    tensor_tree::get_var(var_tree->children[i + 1]->children[1])
                ));
            }

            logprob.push_back(autodiff::logsoftmax(
                autodiff::add(
                    autodiff::mul(feat_t,
                        tensor_tree::get_var(var_tree->children[fc_layer]->children[0])),
                    tensor_tree::get_var(var_tree->children[fc_layer]->children[1])
                )
            ));
        }

        double loss_sum = 0;
        double nframes = 0;

        auto topo_order = autodiff::topo_order(logprob);
        // autodiff::eval(topo_order, autodiff::eval_funcs);

#if DEBUG 
        {
            if (ebt::in(labels[0], ignored)) {
                continue;
            }

            auto& pred = autodiff::get_output<la::tensor_like<double>>(logprob[0]);

            la::tensor<double> gold;
            gold.resize({(unsigned int) label_id.size()});
            gold({label_id.at(labels[0])}) = 1;

            nn::log_loss loss { gold, pred };

            auto param2 = tensor_tree::copy_tree(param);

            auto pre_order = leaves_pre_order(param2);
            auto& t = tensor_tree::get_tensor(pre_order[3]);
            t.data()[0] += 1e-8;

            autodiff::computation_graph graph2;
            auto var_tree2 = tensor_tree::make_var_tree(graph2, param2);

            std::shared_ptr<autodiff::op_t> input2 = graph2.var(input_tensor);

            cnn::multilayer_transcriber trans2;

            for (int i = 0; i < layer; ++i) {
                if (ebt::in(std::string("dropout"), args)) {
                    trans2.layers.push_back(std::make_shared<cnn::dropout_transcriber>(
                        cnn::dropout_transcriber {
                            std::make_shared<cnn::cnn_transcriber>(cnn::cnn_transcriber{}),
                            dropout, gen
                        }
                    ));
                } else {
                    trans2.layers.push_back(std::make_shared<cnn::cnn_transcriber>(
                        cnn::cnn_transcriber{}));
                }
            }

            std::shared_ptr<autodiff::op_t> feat2 = trans(input2, var_tree2->children[0]);

            auto& t2 = tensor_tree::get_tensor(param2->children[0]->children.back()->children.back());

            feat2 = autodiff::reshape(feat2,
                { (unsigned int) frames.size(), (unsigned int) frames.front().size() * t2.size(0) });

            std::vector<std::shared_ptr<autodiff::op_t>> logprob2;
            for (int t = 0; t < frames.size(); ++t) {
                logprob2.push_back(autodiff::logsoftmax(autodiff::add(
                    tensor_tree::get_var(var_tree2->children[2]),
                    autodiff::mul(autodiff::row_at(feat2, t), tensor_tree::get_var(var_tree2->children[1]))
                )));
            }

            auto topo_order2 = autodiff::topo_order(logprob2);
            // autodiff::eval(topo_order2, autodiff::eval_funcs);

            auto& pred2 = autodiff::get_output<la::tensor_like<double>>(logprob2[0]);

            nn::log_loss loss2 { gold, pred2 };

            std::cout << "numeric grad: " << (loss2.loss() - loss.loss()) / 1e-8 << std::endl;

            logprob[0]->grad = std::make_shared<la::tensor<double>>(loss.grad());
            autodiff::grad(logprob[0], autodiff::grad_funcs);

            std::shared_ptr<tensor_tree::vertex> grad = make_tensor_tree(conv_layer, fc_layer);
            tensor_tree::copy_grad(grad, var_tree);
            auto grad_pre_order = leaves_pre_order(grad);

            auto& g_t = tensor_tree::get_tensor(grad_pre_order[3]);
            std::cout << "analytic grad: " << g_t.data()[0] << std::endl;

            continue;
        }
#endif

        for (int t = 0; t < frames.size(); ++t) {
            auto& pred = autodiff::get_output<la::tensor_like<double>>(logprob[t]);

            la::tensor<double> gold;
            gold.resize({(unsigned int) label_id.size()});

            if (!ebt::in(labels[t], ignored)) {
                gold({label_id.at(labels[t])}) = 1;
                nframes += 1;
            }

            nn::log_loss loss { gold, pred };

            if (std::isnan(loss.loss())) {
                std::cerr << "loss is nan" << std::endl;
                exit(1);
            } else {
                loss_sum += loss.loss();
            }

            logprob[t]->grad = std::make_shared<la::tensor<double>>(loss.grad());
        }

        autodiff::grad(topo_order, autodiff::grad_funcs);

        std::cout << "loss: " << loss_sum / nframes << std::endl;

        std::shared_ptr<tensor_tree::vertex> grad = make_tensor_tree(conv_layer, fc_layer);
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
        la::tensor<double> const& v = tensor_tree::get_tensor(vars.front());

        double v1 = v.data()[0];

        opt->update(grad);

        double v2 = v.data()[0];

        std::cout << "weight: " << v1 << " update: " << v2 - v1 << " rate: " << (v2 - v1) / v1 << std::endl;

        std::cout << std::endl;

#if DEBUG_TOP
        if (nsample == DEBUG_TOP) {
            break;
        }
#endif

        ++nsample;
    }

    std::ofstream param_ofs { output_param };
    param_ofs << conv_layer << " " << fc_layer << std::endl;
    tensor_tree::save_tensor(param, param_ofs);
    param_ofs.close();

    std::ofstream opt_data_ofs { output_opt_data };
    opt_data_ofs << conv_layer << " " << fc_layer << std::endl;
    opt->save_opt_data(opt_data_ofs);
    opt_data_ofs.close();
}

