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

struct vae_t {

    std::shared_ptr<autodiff::op_t> mu_e;
    std::shared_ptr<autodiff::op_t> log_sigma_e;

    std::vector<std::shared_ptr<autodiff::op_t>> hidden;

    std::shared_ptr<autodiff::op_t> z;
    std::shared_ptr<autodiff::op_t> mu_d;
    std::shared_ptr<autodiff::op_t> log_sigma_d;

};

vae_t make_vae(std::shared_ptr<autodiff::op_t> input,
    std::shared_ptr<tensor_tree::vertex> var_tree,
    std::default_random_engine &gen)
{
    vae_t vae;

    std::shared_ptr<autodiff::op_t> h = input;

    for (int i = 0; i < var_tree->children[0]->children.size() - 1; ++i) {
        auto a = autodiff::mul(h,
            tensor_tree::get_var(var_tree->children[0]->children[i]->children[0]));
        auto b = autodiff::rep_row_to(
            tensor_tree::get_var(var_tree->children[0]->children[i]->children[1]), a);

        h = autodiff::relu(autodiff::add(a, b));

        vae.hidden.push_back(h);
    }

    auto mu_e_a = autodiff::mul(vae.hidden.back(),
        tensor_tree::get_var(var_tree->children[0]->children.back()->children[0]));
    auto mu_e_b = autodiff::rep_row_to(
        tensor_tree::get_var(var_tree->children[0]->children.back()->children[1]),
        mu_e_a);
    vae.mu_e = autodiff::add(mu_e_a, mu_e_b);

    auto log_sigma_e_a = autodiff::mul(vae.hidden.back(),
        tensor_tree::get_var(var_tree->children[0]->children.back()->children[2]));
    auto log_sigma_e_b = autodiff::rep_row_to(
        tensor_tree::get_var(var_tree->children[0]->children.back()->children[3]),
        log_sigma_e_a);
    vae.log_sigma_e = autodiff::add(log_sigma_e_a, log_sigma_e_b);

    auto eps = autodiff::normal(vae.mu_e, gen);

    auto z = autodiff::add(vae.mu_e, autodiff::emul(eps, autodiff::exp(vae.log_sigma_e)));

    for (int i = 0; i < var_tree->children[1]->children.size() - 1; ++i) {
        auto a = autodiff::mul(z, tensor_tree::get_var(
            var_tree->children[1]->children[i]->children[0]));
        auto b = autodiff::rep_row_to(tensor_tree::get_var(
            var_tree->children[1]->children[i]->children[1]), a);

        h = autodiff::relu(autodiff::add(a, b));

        vae.hidden.push_back(h);
    }

    auto mu_d_a = autodiff::mul(vae.hidden.back(),
        tensor_tree::get_var(var_tree->children[1]->children.back()->children[0]));
    auto mu_d_b = autodiff::rep_row_to(
        tensor_tree::get_var(var_tree->children[1]->children.back()->children[1]),
        mu_d_a);
    vae.mu_d = autodiff::add(mu_d_a, mu_d_b);

    return vae;
}

std::shared_ptr<tensor_tree::vertex> make_tensor_tree(int layer)
{
    tensor_tree::vertex root { "nil" };

    tensor_tree::vertex encoder { "nil" };

    for (int i = 0; i < layer - 1; ++i) {
        tensor_tree::vertex ell { "nil" };
        ell.children.push_back(tensor_tree::make_tensor("encoder weight"));
        ell.children.push_back(tensor_tree::make_tensor("encoder bias"));
        encoder.children.push_back(std::make_shared<tensor_tree::vertex>(ell));
    }

    tensor_tree::vertex enc_mean_var { "nil" };
    enc_mean_var.children.push_back(tensor_tree::make_tensor("encoder mean weight"));
    enc_mean_var.children.push_back(tensor_tree::make_tensor("encoder mean bias"));
    enc_mean_var.children.push_back(tensor_tree::make_tensor("encoder variance weight"));
    enc_mean_var.children.push_back(tensor_tree::make_tensor("encoder variance bias"));
    encoder.children.push_back(std::make_shared<tensor_tree::vertex>(enc_mean_var));

    root.children.push_back(std::make_shared<tensor_tree::vertex>(encoder));

    tensor_tree::vertex decoder { "nil" };

    for (int i = 0; i < layer - 1; ++i) {
        tensor_tree::vertex ell { "nil" };
        ell.children.push_back(tensor_tree::make_tensor("decoder weight"));
        ell.children.push_back(tensor_tree::make_tensor("decoder bias"));
        decoder.children.push_back(std::make_shared<tensor_tree::vertex>(ell));
    }

    tensor_tree::vertex dec_mean_var { "nil" };
    dec_mean_var.children.push_back(tensor_tree::make_tensor("decoder mean weight"));
    dec_mean_var.children.push_back(tensor_tree::make_tensor("decoder mean bias"));
    decoder.children.push_back(std::make_shared<tensor_tree::vertex>(dec_mean_var));

    root.children.push_back(std::make_shared<tensor_tree::vertex>(decoder));

    return std::make_shared<tensor_tree::vertex>(root);
}

struct elbo {

    vae_t vae;
    std::shared_ptr<autodiff::op_t> input;
    int d_x;
    int d_z;

    std::vector<std::shared_ptr<autodiff::op_t>> terms;

    elbo(std::shared_ptr<autodiff::op_t> input,
        vae_t vae, int d_x, int d_z);

    double loss();

    void grad();

};

elbo::elbo(std::shared_ptr<autodiff::op_t> input,
    vae_t vae, int d_x, int d_z)
    : input(input), vae(vae), d_x(d_x), d_z(d_z)
{
    auto sigma_e = autodiff::exp(vae.log_sigma_e);

    auto res = autodiff::sub(input, vae.mu_d);

    terms.push_back(autodiff::dot(res, res));
    terms.push_back(autodiff::dot(vae.mu_e, vae.mu_e));
    terms.push_back(autodiff::dot(sigma_e, sigma_e));
    terms.push_back(autodiff::sum(vae.log_sigma_e));
}

double elbo::loss()
{
    double pi = 3.1415926;

    double neg_logp = 0.5 * d_x * std::log(2 * pi)
        + 0.5 * autodiff::get_output<double>(terms[0]);

    double kl = - autodiff::get_output<double>(terms[3])
        - 0.5 * d_z + 0.5 * autodiff::get_output<double>(terms[1])
        + 0.5 * autodiff::get_output<double>(terms[2]);

    std::cout << "-logp(x|z): " << neg_logp << std::endl;
    std::cout << "KL: " << kl << std::endl;
    std::cout << "d_x: " << d_x << std::endl;
    std::cout << "d_z: " << d_z << std::endl;

    if (neg_logp < 0 || kl < 0) {
        std::cout << "WARNING: -logp or KL is probably wrong" << std::endl;
    }

    return neg_logp + kl;
}

void elbo::grad()
{
    terms[0]->grad = std::make_shared<double>(0.5);
    terms[1]->grad = std::make_shared<double>(0.5);
    terms[2]->grad = std::make_shared<double>(0.5);
    terms[3]->grad = std::make_shared<double>(-1);
}

struct learning_env {

    batch::scp input_scp;

    int layer;
    std::shared_ptr<tensor_tree::vertex> param;

    int d_x;
    int d_z;

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
        "fc-vae",
        "Train a 2-layer VAE",
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

    d_x = tensor_tree::get_tensor(param->children[0]->children[0]->children[0]).size(0);
    d_z = tensor_tree::get_tensor(param->children[0]->children.back()->children[0]).size(0);

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

        vae_t vae = make_vae(input, var_tree, gen);

        elbo loss { input, vae, d_x, d_z };

        double ell = loss.loss();

        if (std::isinf(ell) || std::isnan(ell)) {
            std::cout << "WARNING: loss is inf or nan" << std::endl;
            // exit(1);
            ++nsample;
            continue;
        }

        std::cout << "loss: " << ell << std::endl;

        loss.grad();

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

