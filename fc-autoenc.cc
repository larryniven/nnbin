#include "la/la.h"
#include "autodiff/autodiff.h"
#include "ebt/ebt.h"
#include "util/batch.h"
#include "util/util.h"
#include <fstream>
#include <vector>
#include "nn/nn.h"
#include <random>
#include "nn/tensor-tree.h"
#include <algorithm>

std::vector<std::shared_ptr<autodiff::op_t>> make_autoenc(std::shared_ptr<autodiff::op_t> input,
    std::shared_ptr<tensor_tree::vertex> var_tree)
{
    std::vector<std::shared_ptr<autodiff::op_t>> result;

    std::shared_ptr<autodiff::op_t> h = input;

    for (int i = 0; i < var_tree->children[0]->children.size(); ++i) {
        auto z = autodiff::mul(h, tensor_tree::get_var(
            var_tree->children[0]->children[i]->children[0]));
        auto b = autodiff::rep_row_to(tensor_tree::get_var(
            var_tree->children[0]->children[i]->children[1]), z);
        h = autodiff::relu(autodiff::add(z, b));

        result.push_back(h);
    }

    for (int i = 0; i < var_tree->children[1]->children.size(); ++i) {
        auto z = autodiff::mul(h, tensor_tree::get_var(
            var_tree->children[1]->children[i]->children[0]));
        auto b = autodiff::rep_row_to(tensor_tree::get_var(
            var_tree->children[1]->children[i]->children[1]), z);

        h = autodiff::add(z, b);

        if (i != var_tree->children[1]->children.size() - 1) {
            h = autodiff::relu(h);
        }

        result.push_back(h);
    }

    return result;
}

std::shared_ptr<tensor_tree::vertex> make_tensor_tree(int layer)
{
    tensor_tree::vertex root { "nil" };

    tensor_tree::vertex encoder { "nil" };
    for (int i = 0; i < layer; ++i) {
        tensor_tree::vertex ell { "nil" };
        ell.children.push_back(tensor_tree::make_tensor("encoder weight"));
        ell.children.push_back(tensor_tree::make_tensor("encoder bias"));
        encoder.children.push_back(std::make_shared<tensor_tree::vertex>(ell));
    }
    root.children.push_back(std::make_shared<tensor_tree::vertex>(encoder));

    tensor_tree::vertex decoder { "nil" };
    for (int i = 0; i < layer; ++i) {
        tensor_tree::vertex ell { "nil" };
        ell.children.push_back(tensor_tree::make_tensor("decoder weight"));
        ell.children.push_back(tensor_tree::make_tensor("decoder bias"));
        decoder.children.push_back(std::make_shared<tensor_tree::vertex>(ell));
    }
    root.children.push_back(std::make_shared<tensor_tree::vertex>(decoder));

    return std::make_shared<tensor_tree::vertex>(root);
}

struct learning_env {

    batch::scp input_scp;

    int layer;
    std::shared_ptr<tensor_tree::vertex> param;

    std::shared_ptr<tensor_tree::optimizer> opt;

    double step_size;
    double decay;

    double dropout;
    int seed;

    std::string output_param;
    std::string output_opt_data;

    double clip;

    std::default_random_engine gen;

    std::vector<int> indices;

    std::unordered_map<std::string, std::string> args;

    learning_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "fc-autoenc",
        "Train a 1-layer NN autoencoder",
        {
            {"input-scp", "", true},
            {"param", "", true},
            {"opt-data", "", true},
            {"output-param", "", false},
            {"output-opt-data", "", false},
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
    input_scp.open(args.at("input-scp"));

    std::ifstream param_ifs { args.at("param") };
    if (!param_ifs) {
        throw std::logic_error("failed to open " + args.at("param"));
    }
    std::string line;
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
    if (!opt_data_ifs) {
        throw std::logic_error("failed to open " + args.at("opt-data"));
    }

    opt->load_opt_data(opt_data_ifs);
    opt_data_ifs.close();

    gen = std::default_random_engine { seed };

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

    while (nsample < indices.size()) {
        std::vector<double> raw_input = batch::load_feats(input_scp.at(indices.at(nsample)));

        std::cout << "sample: " << nsample << std::endl;

        autodiff::computation_graph graph;
        std::shared_ptr<tensor_tree::vertex> var_tree = tensor_tree::make_var_tree(graph, param);

        la::cpu::tensor<double> input_tensor { la::cpu::vector<double> { raw_input } };

        std::shared_ptr<autodiff::op_t> input = graph.var(input_tensor);

        auto layers = make_autoenc(input, var_tree);

        auto& pred = autodiff::get_output<la::cpu::tensor_like<double>>(layers.back());

        la::cpu::weak_tensor<double> gold { input_tensor.data(), input_tensor.sizes() };

        nn::l2_loss loss { gold, pred };

        layers.back()->grad = std::make_shared<la::cpu::tensor<double>>(loss.grad());
        
        double ell = loss.loss();

        if (std::isnan(ell)) {
            std::cerr << "loss is nan" << std::endl;
            exit(1);
        }

        std::cout << "loss: " << ell << std::endl;

        auto topo_order = autodiff::natural_topo_order(graph);
        autodiff::guarded_grad(topo_order, autodiff::grad_funcs);

        std::shared_ptr<tensor_tree::vertex> grad = make_tensor_tree(layer);
        tensor_tree::copy_grad(grad, var_tree);

        double n = tensor_tree::norm(grad);

        std::cout << "grad norm: " << n << std::endl;

        if (ebt::in(std::string("clip"), args)) {
            if (n > clip) {
                tensor_tree::axpy(grad, clip / n, grad);
                std::cout << "gradient clipped" << std::endl;
            }
        }

        std::vector<std::shared_ptr<tensor_tree::vertex>> vars = tensor_tree::leaves_pre_order(param);
        la::cpu::tensor<double> const& v = tensor_tree::get_tensor(vars.front());

        double v1 = v.data()[0];

        opt->update(grad);

        double v2 = v.data()[0];

        std::cout << vars.front()->name << " weight: " << v1
            << " update: " << v2 - v1 << " rate: " << (v2 - v1) / v1 << std::endl;

        std::cout << std::endl;

        ++nsample;
    }

    std::ofstream param_ofs { output_param };
    param_ofs << layer << std::endl;
    tensor_tree::save_tensor(param, param_ofs);
    param_ofs.close();

    std::ofstream opt_data_ofs { output_opt_data };
    opt->save_opt_data(opt_data_ofs);
    opt_data_ofs.close();
}

