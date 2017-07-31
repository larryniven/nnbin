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
#include "nn/lstm-tensor-tree.h"
#include <algorithm>

std::shared_ptr<tensor_tree::vertex> make_tensor_tree(int layer)
{
    tensor_tree::vertex root { "nil" };

    lstm::multilayer_lstm_tensor_tree_factory frame_lstm_factory {
        std::make_shared<lstm::bi_lstm_tensor_tree_factory>(
        lstm::bi_lstm_tensor_tree_factory {
            std::make_shared<lstm::lstm_tensor_tree_factory>(
                lstm::lstm_tensor_tree_factory{})
        }),
        layer
    };

    root.children.push_back(frame_lstm_factory());

    tensor_tree::vertex seg { "nil" };
    seg.children.push_back(tensor_tree::make_tensor("left"));
    seg.children.push_back(tensor_tree::make_tensor("center"));
    seg.children.push_back(tensor_tree::make_tensor("right"));
    seg.children.push_back(tensor_tree::make_tensor("bias"));

    root.children.push_back(std::make_shared<tensor_tree::vertex>(seg));

    lstm::multilayer_lstm_tensor_tree_factory seg_lstm_factory {
        std::make_shared<lstm::bi_lstm_tensor_tree_factory>(
        lstm::bi_lstm_tensor_tree_factory {
            std::make_shared<lstm::lstm_tensor_tree_factory>(
                lstm::lstm_tensor_tree_factory{})
        }),
        1
    };

    root.children.push_back(seg_lstm_factory());

    tensor_tree::vertex fc { "nil" };
    fc.children.push_back(tensor_tree::make_tensor("softmax"));
    fc.children.push_back(tensor_tree::make_tensor("bias"));

    root.children.push_back(std::make_shared<tensor_tree::vertex>(fc));

    return std::make_shared<tensor_tree::vertex>(root);
}

lstm::trans_seq_t
transcribe_seg_feat(std::shared_ptr<tensor_tree::vertex> var_tree,
    lstm::trans_seq_t const& seq,
    std::vector<speech::segment> const& segs,
    int output_dim)
{
    auto& comp_graph = *tensor_tree::get_var(var_tree->children[0])->graph;

    auto left_feat = autodiff::mul(seq.feat, tensor_tree::get_var(var_tree->children[0]));
    auto center_feat = autodiff::mul(seq.feat, tensor_tree::get_var(var_tree->children[1]));
    auto right_feat = autodiff::mul(seq.feat, tensor_tree::get_var(var_tree->children[2]));

    la::cpu::tensor<double> output_tensor;
    output_tensor.resize({(unsigned int) segs.size(), (unsigned int) output_dim});
    auto output_storage = comp_graph.var(output_tensor);

    std::vector<std::shared_ptr<autodiff::op_t>> outputs;

    for (int i = 0; i < segs.size(); ++i) {
        auto output_i = autodiff::weak_var(output_storage, i * output_dim,
            std::vector<unsigned int> { (unsigned int) output_dim });

        int start_time = std::min<int>(segs[i].start_time, seq.nframes - 1);
        int end_time = std::min<int>(segs[i].end_time, seq.nframes);

        auto output = autodiff::add(output_i, std::vector<std::shared_ptr<autodiff::op_t>> {
            autodiff::row_at(left_feat, start_time),
            autodiff::row_at(center_feat, (start_time + end_time - 1) / 2),
            autodiff::row_at(right_feat, end_time - 1)
        });

        outputs.push_back(output);
    }

    auto output_tmp = autodiff::weak_cat(outputs, output_storage);

    lstm::trans_seq_t result;
    result.batch_size = 1;
    result.nframes = segs.size();
    result.dim = output_dim;

    result.feat = autodiff::add(output_tmp,
        autodiff::rep_row_to(tensor_tree::get_var(var_tree->children[3]), output_tmp));

    return result;
}

struct learning_env {

    speech::batch_indices frame_batch;
    speech::batch_indices seg_batch;

    int layer;
    std::shared_ptr<tensor_tree::vertex> param;

    std::shared_ptr<tensor_tree::optimizer> opt;

    std::string output_param;
    std::string output_opt_data;

    int seed;
    double dropout;
    double step_size;
    double clip;

    int output_dim;

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
        "seg-lstm-learn",
        "Train a LSTM segment classifier",
        {
            {"frame-batch", "", true},
            {"seg-batch", "", true},
            {"param", "", true},
            {"opt-data", "", true},
            {"output-param", "", false},
            {"output-opt-data", "", false},
            {"label", "", true},
            {"ignore", "", false},
            {"dropout", "", false},
            {"shuffle", "", false},
            {"seed", "", false},
            {"seg-lstm", "", false},
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
    frame_batch.open(args.at("frame-batch"));
    seg_batch.open(args.at("seg-batch"));

    std::string line;

    std::ifstream param_ifs { args.at("param") };
    std::getline(param_ifs, line);
    layer = std::stoi(line);
    param = make_tensor_tree(layer);
    tensor_tree::load_tensor(param, param_ifs);
    param_ifs.close();

    output_dim = tensor_tree::get_tensor(param->children[1]->children[0]).size(1);

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

        pos = seg_batch.pos;
        for (int i = 0; i < indices.size(); ++i) {
            seg_batch.pos[i] = pos[indices[i]];
        }
    }
}

void learning_env::run()
{
    ebt::Timer timer;

    int nsample = 0;

    std::shared_ptr<tensor_tree::vertex> accu_grad = tensor_tree::deep_copy(param);
    tensor_tree::zero(accu_grad);

    while (nsample < frame_batch.pos.size()) {
        std::cout << "sample: " << nsample << std::endl;
        std::cout << "index: " << indices[nsample] << std::endl;

        std::vector<std::vector<double>> frames = speech::load_frame_batch(frame_batch.at(nsample));

	std::vector<speech::segment> segs = speech::load_segment_batch(seg_batch.at(nsample));

        std::cout << "frames: " << frames.size() << std::endl;
        std::cout << "segs: " << segs.size() << std::endl;

        autodiff::computation_graph graph;

        std::vector<double> input_cat;
        input_cat.reserve(frames.size() * frames.front().size());

        for (int i = 0; i < frames.size(); ++i) {
            input_cat.insert(input_cat.end(), frames[i].begin(), frames[i].end());
        }

        std::shared_ptr<autodiff::op_t> input = graph.var(
            la::cpu::tensor<double>(la::cpu::weak_vector<double>(input_cat.data(), input_cat.size()),
            { (unsigned int) frames.size(), (unsigned int) frames.front().size() }));

        input->grad_needed = false;

        std::shared_ptr<tensor_tree::vertex> var_tree = tensor_tree::make_var_tree(graph, param);

        lstm::trans_seq_t input_seq;
        input_seq.nframes = frames.size();
        input_seq.batch_size = batch_size;
        input_seq.dim = frames.front().size();
        input_seq.feat = input;
        input_seq.mask = nullptr;

        std::shared_ptr<lstm::transcriber> trans;
        trans = lstm_frame::make_transcriber(param->children[0], dropout, &gen, false);
        lstm::trans_seq_t frame_feat = (*trans)(var_tree->children[0], input_seq);

        auto seg_feat = transcribe_seg_feat(var_tree->children[1], frame_feat, segs, output_dim);

        if (ebt::in(std::string("seg-lstm"), args)) {
            trans = lstm_frame::make_transcriber(param->children[2], dropout, &gen, false);
            seg_feat = (*trans)(var_tree->children[2], seg_feat);
        }

        lstm::fc_transcriber fc_trans { (int) label_id.size() };
        lstm::logsoftmax_transcriber logsoftmax_trans;
        auto score_seq = fc_trans(var_tree->children[3], seg_feat);
        auto output_seq = logsoftmax_trans(nullptr, score_seq);

        std::shared_ptr<autodiff::op_t> logprob = output_seq.feat;

        std::vector<double> gold_vec;
        gold_vec.resize(segs.size() * label_id.size());
        int nsegs= 0;

        for (int t = 0; t < segs.size(); ++t) {
            if (!ebt::in(segs[t].label, ignored)) {
                gold_vec[t * label_id.size() + label_id.at(segs[t].label)] = 1;
                nsegs += 1;
            }
        }

        la::cpu::weak_tensor<double> gold { gold_vec.data(),
            {(unsigned int) segs.size(), (unsigned int) label_id.size()}};
        auto& pred = autodiff::get_output<la::cpu::tensor_like<double>>(logprob);
        nn::log_loss loss { gold, pred };

        logprob->grad = std::make_shared<la::cpu::tensor<double>>(loss.grad());
        
        double ell = loss.loss();

        if (std::isnan(ell)) {
            throw std::logic_error("loss is nan");
        }

        std::cout << "loss: " << ell / batch_size << std::endl;
        std::cout << "E: " << ell / nsegs << std::endl;

        auto topo_order = autodiff::natural_topo_order(graph);
        autodiff::guarded_grad(topo_order, autodiff::grad_funcs);

#if 0
        auto& t = autodiff::get_grad<la::cpu::tensor_like<double>>(trans1->debug);
        for (int i = 0; i < t.size(0); ++i) {
            for (int j = 0; j < t.size(1); ++j) {
                std::cout << t({i, j}) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
#endif

        std::shared_ptr<tensor_tree::vertex> grad;
        grad = make_tensor_tree(layer);
        tensor_tree::copy_grad(grad, var_tree);

        tensor_tree::iadd(accu_grad, grad);

        if (nsample % batch_size == batch_size - 1) {

            tensor_tree::imul(accu_grad, 1.0 / batch_size);

            double n = tensor_tree::norm(accu_grad);

            std::cout << "grad norm: " << n << std::endl;

            if (ebt::in(std::string("clip"), args)) {
                if (n > clip) {
                    tensor_tree::imul(accu_grad, clip / n);
                    std::cout << "gradient clipped" << std::endl;
                }
            }

            std::vector<std::shared_ptr<tensor_tree::vertex>> vars = tensor_tree::leaves_pre_order(param);

            la::cpu::tensor<double> const& v = tensor_tree::get_tensor(vars.front());

            double v1 = v.data()[0];

            opt->update(accu_grad);

            tensor_tree::zero(accu_grad);

            double v2 = v.data()[0];

            std::cout << "name: " << vars.front()->name << " weight: " << v1 << " update: " << v2 - v1 << " rate: " << (v2 - v1) / v1 << std::endl;

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

