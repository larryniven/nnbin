#include "la/la-gpu.h"
#include "autodiff/autodiff-gpu.h"
#include "nn/tensor-tree-gpu.h"
#include "nn/nn-gpu.h"
#include "ebt/ebt.h"
#include "util/speech.h"
#include "util/batch.h"
#include "util/util.h"
#include <fstream>
#include <vector>
#include "nn/lstm.h"
#include "nn/lstm-tensor-tree.h"
#include <random>
#include "nn/lstm-frame.h"
#include <algorithm>

#if 0
std::tuple<autodiff::computation_graph,
    std::shared_ptr<tensor_tree::vertex>>
cpu_comp_graph(
    std::vector<double> const& input_cat,
    std::vector<std::vector<double>> const& frames,
    std::shared_ptr<tensor_tree::vertex> param,
    std::vector<std::string> const& labels,
    std::unordered_map<std::string, int> const& label_id,
    std::unordered_set<std::string> const& ignored);

void check_output_equality(
    std::vector<std::shared_ptr<autodiff::op_t>> const& gpu_vars,
    std::vector<std::shared_ptr<autodiff::op_t>> const& cpu_vars);

void check_grad_equality(
    std::vector<std::shared_ptr<autodiff::op_t>> const& gpu_vars,
    std::vector<std::shared_ptr<autodiff::op_t>> const& cpu_vars);

void check_tensor_equality(
    std::shared_ptr<tensor_tree::vertex> param,
    std::shared_ptr<tensor_tree::vertex> cpu_param);

std::shared_ptr<tensor_tree::vertex> make_tensor_tree()
{
    tensor_tree::vertex root { "nil" };
    root.children.push_back(lstm::lstm_tensor_tree_factory{}());
    tensor_tree::vertex fc { "nil" };
    fc.children.push_back(tensor_tree::make_tensor("softmax weight"));
    fc.children.push_back(tensor_tree::make_tensor("softmax bias"));
    root.children.push_back(std::make_shared<tensor_tree::vertex>(fc));

    return std::make_shared<tensor_tree::vertex>(root);
}
#endif

struct learning_env {

    batch::scp frame_scp;
    batch::scp label_scp;

    int layer;
    std::shared_ptr<tensor_tree::vertex> param;
    std::shared_ptr<tensor_tree::vertex> cpu_param;

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

    std::unordered_map<std::string, int> label_id;

    std::unordered_set<std::string> ignored;

    std::unordered_map<std::string, std::string> args;

    learning_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "frame-lstm-learn-gpu",
        "Train a LSTM frame classifier",
        {
            {"frame-scp", "", true},
            {"label-scp", "", true},
            {"param", "", true},
            {"opt-data", "", true},
            {"output-param", "", false},
            {"output-opt-data", "", false},
            {"label", "", true},
            {"ignore", "", false},
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
    frame_scp.open(args.at("frame-scp"));
    label_scp.open(args.at("label-scp"));

    std::string line;

    std::ifstream param_ifs { args.at("param") };
    std::getline(param_ifs, line);
    layer = std::stoi(line);

    param = lstm_frame::make_tensor_tree(layer);

    // param = make_tensor_tree();

    tensor_tree::load_tensor(param, param_ifs);
    param_ifs.close();

    // cpu_param = tensor_tree::deep_copy(param);

    tensor_tree::gpu::to_device(param);

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

    std::vector<std::string> label_vec = util::load_label_set(args.at("label"));
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

    batch_size = 1;
    if (ebt::in(std::string("batch-size"), args)) {
        batch_size = std::stoi(args.at("batch-size"));
    }

    if (args.at("opt") == "rmsprop") {
        double decay = std::stod(args.at("decay"));
        opt = std::make_shared<tensor_tree::gpu::rmsprop_opt>(
            tensor_tree::gpu::rmsprop_opt(param, step_size, decay));
    } else if (args.at("opt") == "const-step") {
        opt = std::make_shared<tensor_tree::gpu::const_step_opt>(
            tensor_tree::gpu::const_step_opt(param, step_size));
    } else if (args.at("opt") == "adagrad") {
        opt = std::make_shared<tensor_tree::gpu::adagrad_opt>(
            tensor_tree::gpu::adagrad_opt(param, step_size));
    } else {
        throw std::logic_error("unknown optimizer " + args.at("opt"));
    }

    gen = std::default_random_engine { seed };

    std::ifstream opt_data_ifs { args.at("opt-data") };
    opt->load_opt_data(opt_data_ifs);
    opt_data_ifs.close();

    indices.resize(frame_scp.entries.size());

    for (int i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }

    if (ebt::in(std::string("shuffle"), args)) {
        std::shuffle(indices.begin(), indices.end(), gen);
    }
}

void learning_env::run()
{
    ebt::accu_timer<0> timer;

    int nsample = 0;

    std::shared_ptr<tensor_tree::vertex> accu_grad = tensor_tree::gpu::deep_copy(param);
    tensor_tree::gpu::zero(accu_grad);

    while (nsample < indices.size()) {
        std::cout << "sample: " << nsample << std::endl;
        std::cout << "index: " << indices[nsample] << std::endl;

        std::vector<std::vector<double>> frames = speech::load_frame_batch(
            frame_scp.at(indices[nsample]));

	std::vector<std::string> labels = speech::load_label_seq_batch(
            label_scp.at(indices[nsample]));

        std::cout << "frames: " << frames.size() << std::endl;
        std::cout << "labels: " << labels.size() << std::endl;

        assert(frames.size() == labels.size());

        autodiff::computation_graph graph;
        graph.eval_funcs = autodiff::gpu::eval_funcs;
        graph.grad_funcs = autodiff::gpu::grad_funcs;

        std::vector<double> input_cat;
        input_cat.reserve(frames.size() * frames.front().size());

        for (int i = 0; i < frames.size(); ++i) {
            input_cat.insert(input_cat.end(), frames[i].begin(), frames[i].end());
        }

        std::shared_ptr<autodiff::op_t> input = graph.var(
            la::gpu::tensor<double>(la::cpu::weak_tensor<double>(input_cat.data(),
            { (unsigned int) frames.size(), (unsigned int) frames.front().size() })));

        input->grad_needed = false;

        std::shared_ptr<tensor_tree::vertex> var_tree = tensor_tree::make_var_tree(graph, param);

        lstm::trans_seq_t input_seq;
        input_seq.nframes = frames.size();
        input_seq.batch_size = batch_size;
        input_seq.dim = frames.front().size();
        input_seq.feat = input;
        input_seq.mask = nullptr;

        std::shared_ptr<lstm::transcriber> trans = lstm_frame::make_transcriber(
            param->children[0], dropout, &gen, false);

        lstm::trans_seq_t feat_seq = (*trans)(var_tree->children[0], input_seq);

        // lstm::lstm_transcriber trans { 128, false };
        // lstm::trans_seq_t feat_seq = trans(var_tree->children[0], input_seq);

        lstm::fc_transcriber fc_trans { (int) label_id.size() };
        lstm::logsoftmax_transcriber logsoftmax_trans;
        auto score_seq = fc_trans(var_tree->children[1], feat_seq);
        auto output_seq = logsoftmax_trans(nullptr, score_seq);

        std::shared_ptr<autodiff::op_t> logprob = output_seq.feat;

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
        la::gpu::tensor<double> dgold { gold };
        auto& dpred = autodiff::get_output<la::gpu::tensor_like<double>>(logprob);
        nn::gpu::log_loss loss { dgold, dpred };

        logprob->grad = std::make_shared<la::gpu::tensor<double>>(loss.grad());

        double ell = loss.loss();

        if (std::isnan(ell)) {
            throw std::logic_error("loss is nan");
        }

        std::cout << "loss: " << ell / batch_size << std::endl;
        std::cout << "E: " << ell / nframes << std::endl;

        // autodiff::computation_graph cpu_graph;
        // std::shared_ptr<tensor_tree::vertex> cpu_var_tree;
        // std::tie(cpu_graph, cpu_var_tree) = cpu_comp_graph(
        //     input_cat, frames, cpu_param, labels, label_id, ignored);

        // auto cpu_topo_order = autodiff::natural_topo_order(cpu_graph);

        auto topo_order = autodiff::natural_topo_order(graph);

        // check_output_equality(topo_order, cpu_topo_order);

        // autodiff::guarded_grad(cpu_topo_order, autodiff::grad_funcs);

        autodiff::guarded_grad(topo_order, autodiff::gpu::grad_funcs);

        // check_grad_equality(topo_order, cpu_topo_order);

#if 0
        std::shared_ptr<tensor_tree::vertex> cpu_grad = tensor_tree::deep_copy(cpu_param);
        tensor_tree::zero(cpu_grad);

        tensor_tree::copy_grad(cpu_grad, cpu_var_tree);

        double cpu_n = tensor_tree::norm(cpu_grad);

        if (cpu_n > clip) {
            tensor_tree::axpy(cpu_grad, clip / cpu_n - 1, cpu_grad);
        }

        tensor_tree::const_step_update(cpu_param, cpu_grad, step_size);
#endif

        std::shared_ptr<tensor_tree::vertex> grad = tensor_tree::gpu::deep_copy(param);
        tensor_tree::gpu::zero(grad);

        tensor_tree::copy_grad(grad, var_tree);

        tensor_tree::gpu::axpy(accu_grad, 1, grad);

        if (nsample % batch_size == batch_size - 1) {

            tensor_tree::gpu::axpy(accu_grad, 1.0 / batch_size - 1, accu_grad);

            double n = tensor_tree::gpu::norm(accu_grad);

            std::cout << "grad norm: " << n << std::endl;

            if (ebt::in(std::string("clip"), args)) {
                if (n > clip) {
                    tensor_tree::gpu::axpy(accu_grad, clip / n - 1, accu_grad);
                    std::cout << "gradient clipped" << std::endl;
                }
            }

            // check_tensor_equality(accu_grad, cpu_grad);

            std::vector<std::shared_ptr<tensor_tree::vertex>> vars = tensor_tree::leaves_pre_order(param);

            la::gpu::tensor<double> const& v = tensor_tree::gpu::get_gpu_tensor(vars.front());

            la::cpu::tensor<double> hv = la::gpu::to_host(v);

            double v1 = hv.data()[0];

            opt->update(accu_grad);

            tensor_tree::gpu::zero(accu_grad);

            hv = la::gpu::to_host(v);

            double v2 = hv.data()[0];

            std::cout << "name: " << vars.front()->name << " weight: " << v1
                << " update: " << v2 - v1 << " rate: " << (v2 - v1) / v1 << std::endl;

            std::cout << std::endl;

        }

        // check_tensor_equality(param, cpu_param);

        ++nsample;

#if DEBUG_TOP
        if (nsample >= DEBUG_TOP) {
            break;
        }
#endif
    }

    std::ofstream param_ofs { output_param };
    param_ofs << layer << std::endl;
    tensor_tree::gpu::to_host(param);
    tensor_tree::save_tensor(param, param_ofs);
    param_ofs.close();

    std::ofstream opt_data_ofs { output_opt_data };
    opt->save_opt_data(opt_data_ofs);
    opt_data_ofs.close();
}

#if 0
std::tuple<autodiff::computation_graph,
    std::shared_ptr<tensor_tree::vertex>>
cpu_comp_graph(
    std::vector<double> const& input_cat,
    std::vector<std::vector<double>> const& frames,
    std::shared_ptr<tensor_tree::vertex> cpu_param,
    std::vector<std::string> const& labels,
    std::unordered_map<std::string, int> const& label_id,
    std::unordered_set<std::string> const& ignored)
{
    int batch_size = 1;

    autodiff::computation_graph cpu_graph;
    cpu_graph.eval_funcs = autodiff::eval_funcs;
    cpu_graph.grad_funcs = autodiff::grad_funcs;

    std::shared_ptr<autodiff::op_t> input = cpu_graph.var(
        la::cpu::weak_tensor<double>(const_cast<double*>(input_cat.data()),
        { (unsigned int) frames.size(), (unsigned int) frames.front().size() }));

    input->grad_needed = false;

    std::shared_ptr<tensor_tree::vertex> var_tree = tensor_tree::make_var_tree(cpu_graph, cpu_param);

    lstm::lstm_transcriber trans { 128, false };

    lstm::trans_seq_t input_seq;
    input_seq.nframes = frames.size();
    input_seq.batch_size = batch_size;
    input_seq.dim = frames.front().size();
    input_seq.feat = input;
    input_seq.mask = nullptr;

    lstm::trans_seq_t feat_seq = trans(var_tree->children[0], input_seq);

    lstm::fc_transcriber fc_trans { (int) label_id.size() };
    lstm::logsoftmax_transcriber logsoftmax_trans;
    auto score_seq = fc_trans(var_tree->children[1], feat_seq);
    auto output_seq = logsoftmax_trans(nullptr, score_seq);

    std::shared_ptr<autodiff::op_t> logprob = output_seq.feat;

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

    return std::make_tuple(cpu_graph, var_tree);
}

void check_output_equality(
    std::vector<std::shared_ptr<autodiff::op_t>> const& gpu_vars,
    std::vector<std::shared_ptr<autodiff::op_t>> const& cpu_vars)
{
    assert(gpu_vars.size() == cpu_vars.size());

    for (int i = 0; i < gpu_vars.size(); ++i) {
        assert(gpu_vars[i]->name == cpu_vars[i]->name);

        auto& gt = autodiff::get_output<la::gpu::tensor_like<double>>(gpu_vars[i]);
        la::cpu::tensor<double> hgt = la::gpu::to_host(gt);
        auto& ct = autodiff::get_output<la::cpu::tensor_like<double>>(cpu_vars[i]);

        assert(hgt.vec_size() == ct.vec_size());

        for (int j = 0; j < hgt.vec_size(); ++j) {
            assert(hgt.data()[j] == ct.data()[j]);
        }
    }
}

void check_grad_equality(
    std::vector<std::shared_ptr<autodiff::op_t>> const& gpu_vars,
    std::vector<std::shared_ptr<autodiff::op_t>> const& cpu_vars)
{
    assert(gpu_vars.size() == cpu_vars.size());

    for (int i = 0; i < gpu_vars.size(); ++i) {
        assert(gpu_vars[i]->name == cpu_vars[i]->name);

        assert((gpu_vars[i]->grad == nullptr && cpu_vars[i]->grad == nullptr)
            || (gpu_vars[i]->grad != nullptr && cpu_vars[i]->grad != nullptr));

        if (gpu_vars[i]->grad == nullptr) {
            continue;
        }

        auto& gt = autodiff::get_grad<la::gpu::tensor_like<double>>(gpu_vars[i]);
        la::cpu::tensor<double> hgt = la::gpu::to_host(gt);
        auto& ct = autodiff::get_grad<la::cpu::tensor_like<double>>(cpu_vars[i]);

        assert(hgt.vec_size() == ct.vec_size());

        for (int j = 0; j < hgt.vec_size(); ++j) {
            assert(hgt.data()[j] == ct.data()[j]);
        }
    }
}

void check_tensor_equality(
    std::shared_ptr<tensor_tree::vertex> gpu_tree,
    std::shared_ptr<tensor_tree::vertex> cpu_tree)
{
    std::vector<std::shared_ptr<tensor_tree::vertex>> gpu_vars = tensor_tree::leaves_pre_order(gpu_tree);
    std::vector<std::shared_ptr<tensor_tree::vertex>> cpu_vars = tensor_tree::leaves_pre_order(cpu_tree);

    assert(gpu_vars.size() == cpu_vars.size());

    for (int i = 0; i < gpu_vars.size(); ++i) {
        auto& ct = tensor_tree::get_tensor(cpu_vars[i]);
        auto& gt = tensor_tree::gpu::get_gpu_tensor(gpu_vars[i]);
        la::cpu::tensor<double> hgt = la::gpu::to_host(gt);

        assert(hgt.vec_size() == ct.vec_size());

        for (int j = 0; j < hgt.vec_size(); ++j) {
            assert(hgt.data()[j] == ct.data()[j]);
        }
    }
}
#endif
