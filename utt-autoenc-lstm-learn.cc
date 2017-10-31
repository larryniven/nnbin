#include "la/la.h"
#include "autodiff/autodiff.h"
#include "ebt/ebt.h"
#include "speech/speech.h"
#include "nn/nn.h"
#include "nn/tensor-tree.h"
#include "nn/lstm.h"
#include "nn/lstm-tensor-tree.h"
#include <fstream>
#include <vector>
#include <random>
#include <algorithm>

std::shared_ptr<tensor_tree::vertex> make_tensor_tree(int layer)
{
    lstm::multilayer_lstm_tensor_tree_factory lstm_factory {
        std::make_shared<lstm::bi_lstm_tensor_tree_factory>(
            lstm::bi_lstm_tensor_tree_factory {
                std::make_shared<lstm::lstm_tensor_tree_factory>(
                    lstm::lstm_tensor_tree_factory{})
            }),
        layer
    };

    tensor_tree::vertex root { "nil" };

    root.children.push_back(lstm_factory());
    root.children.push_back(lstm_factory());
    root.children.push_back(tensor_tree::make_tensor("regression weight"));
    root.children.push_back(tensor_tree::make_tensor("regression bias"));

    return std::make_shared<tensor_tree::vertex>(root);
}

std::shared_ptr<lstm::transcriber>
make_transcriber(
    std::shared_ptr<tensor_tree::vertex> param,
    double dropout,
    std::default_random_engine *gen,
    bool pyramid)
{
    int layer = param->children.size();

    lstm::layered_transcriber trans;

    for (int i = 0; i < layer; ++i) {
        std::shared_ptr<lstm::transcriber> f_trans;
        std::shared_ptr<lstm::transcriber> b_trans;

        f_trans = std::make_shared<lstm::lstm_transcriber>(
            lstm::lstm_transcriber { (int) tensor_tree::get_tensor(
                param->children[i]->children[0]->children[2]).size(0) });
        b_trans = std::make_shared<lstm::lstm_transcriber>(
            lstm::lstm_transcriber { (int) tensor_tree::get_tensor(
                param->children[i]->children[1]->children[2]).size(0), true });

        if (dropout != 0.0) {
            f_trans = std::make_shared<lstm::input_dropout_transcriber>(
                lstm::input_dropout_transcriber { f_trans, dropout, *gen });

            b_trans = std::make_shared<lstm::input_dropout_transcriber>(
                lstm::input_dropout_transcriber { b_trans, dropout, *gen });
        }

        trans.layer.push_back(std::make_shared<lstm::bi_transcriber>(
            lstm::bi_transcriber { (int) tensor_tree::get_tensor(
                param->children[i]->children[2]).size(1), f_trans, b_trans }));
    }

    return std::make_shared<lstm::layered_transcriber>(trans);
}

struct learning_env {

    speech::scp frame_scp;

    int layer;
    std::shared_ptr<tensor_tree::vertex> param;

    double input_dropout;

    std::string output_param;
    std::string output_opt_data;

    std::shared_ptr<tensor_tree::optimizer> opt;

    double step_size;
    double decay;
    double clip;

    int seed;
    std::default_random_engine gen;

    int save_every;
    std::string save_every_prefix;

    std::vector<int> indices;

    std::unordered_map<std::string, std::string> args;

    learning_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "utt-autoenc-lstm-learn",
        "Train an lstm autoencoder",
        {
            {"frame-scp", "", true},
            {"param", "", true},
            {"opt-data", "", true},
            {"output-param", "", false},
            {"output-opt-data", "", false},
            {"input-dropout", "", false},
            {"seed", "", false},
            {"shuffle", "", false},
            {"opt", "const-step,adagrad,rmsprop,adam", true},
            {"step-size", "", true},
            {"momentum", "", false},
            {"clip", "", false},
            {"decay", "", false},
            {"save-every", "", false},
            {"save-every-prefix", "", false},
            {"loss-mode", "", false},
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
    frame_scp.open(args.at("frame-scp"));

    std::string line;
    std::ifstream param_ifs { args.at("param") };
    std::getline(param_ifs, line);
    layer = std::stoi(line);
    param = make_tensor_tree(layer);
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

    input_dropout = 0;
    if (ebt::in(std::string("input-dropout"), args)) {
        input_dropout = std::stod(args.at("input-dropout"));
    }

    seed = 1;
    if (ebt::in(std::string("seed"), args)) {
        seed = std::stoi(args.at("seed"));
    }

    save_every = -1;
    if (ebt::in(std::string("save-every"), args)) {
        save_every = std::stoi(args.at("save-every"));
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

    for (int i = 0; i < frame_scp.entries.size(); ++i) {
        indices.push_back(i);
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

        std::vector<std::vector<double>> frames = speech::load_frame_batch(
            frame_scp.at(indices.at(nsample)));

        int input_dim = frames.front().size();

        std::vector<double> input_vec;

        for (int i = 0; i < frames.size(); ++i) {
            for (int j = 0; j < input_dim; ++j) {
                input_vec.push_back(frames[i][j]);
            }
        }

        la::cpu::tensor<double> input_t { la::cpu::vector<double>(input_vec),
            { (unsigned int) frames.size(), 1, (unsigned int) input_dim }};

        auto input_var = graph.var(input_t);

        lstm::trans_seq_t seq = lstm::make_trans_seq(input_var);

        std::shared_ptr<lstm::transcriber> enc_trans
            = make_transcriber(param->children[0], input_dropout, &gen, false);

        lstm::trans_seq_t h1 = (*enc_trans)(var_tree->children[0], seq);

        std::shared_ptr<lstm::transcriber> dec_trans
            = make_transcriber(param->children[1], 0.0, nullptr, false);
        
        lstm::trans_seq_t h2 = (*dec_trans)(var_tree->children[1], h1);

        auto z = autodiff::mul(h2.feat, tensor_tree::get_var(var_tree->children[2]));
        auto b = autodiff::rep_row_to(tensor_tree::get_var(var_tree->children[3]), z);
        auto pred = autodiff::add(z, b);

        auto& pred_t = autodiff::get_output<la::cpu::tensor_like<double>>(pred);

        nn::l2_loss loss { input_t, pred_t };

        pred->grad = std::make_shared<la::cpu::tensor<double>>(loss.grad());
        
        double ell = loss.loss();

        if (std::isnan(ell)) {
            std::cerr << "loss is nan" << std::endl;
            exit(1);
        }

        std::cout << "loss: " << ell << std::endl;
        std::cout << "E: " << ell / frames.size() << std::endl;

        if (ebt::in(std::string("loss-mode"), args)) {
            std::cout << std::endl;
            ++nsample;
            continue;
        }

        auto topo_order = autodiff::natural_topo_order(graph);
        autodiff::guarded_grad(topo_order, autodiff::grad_funcs);

        std::shared_ptr<tensor_tree::vertex> grad = make_tensor_tree(layer);
        tensor_tree::copy_grad(grad, var_tree);

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

        if (save_every > 0 && nsample % save_every == 0) {
            std::ofstream param_ofs { args.at("save-every-prefix") + "-" + std::to_string(nsample) };
            tensor_tree::save_tensor(param, param_ofs);
            param_ofs.close();
        }

        ++nsample;

    }

    if (ebt::in(std::string("loss-mode"), args)) {
        return;
    }

    std::ofstream param_ofs { output_param };
    param_ofs << layer << std::endl;
    tensor_tree::save_tensor(param, param_ofs);
    param_ofs.close();

    std::ofstream opt_data_ofs { output_opt_data };
    opt->save_opt_data(opt_data_ofs);
    opt_data_ofs.close();
}

