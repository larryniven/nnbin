#include "speech/speech.h"
#include <fstream>
#include "ebt/ebt.h"
#include "nn/lstm-frame.h"
#include <sstream>
#include "nn/seq2seq.h"
#include <algorithm>
#include "nn/nn.h"

using namespace std::string_literals;

struct learning_env {

    speech::batch_indices frame_batch;
    speech::batch_indices label_batch;

    std::string output_param;
    std::string output_opt_data;

    int layer;
    std::shared_ptr<tensor_tree::vertex> param;
    std::shared_ptr<tensor_tree::vertex> opt_data;

    double step_size;
    double dropout;
    double clip;

    int seed;
    std::default_random_engine gen;

    std::unordered_map<std::string, int> label_id;
    std::vector<std::string> id_label;

    std::vector<int> indices;

    std::shared_ptr<tensor_tree::optimizer> opt;

    std::unordered_map<std::string, std::string> args;

    learning_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "seq2seq-learn",
        "Train a seq2seq model",
        {
            {"frame-batch", "", false},
            {"label-batch", "", true},
            {"param", "", true},
            {"opt-data", "", true},
            {"output-param", "", false},
            {"output-opt-data", "", false},
            {"label", "", true},
            {"dropout", "", false},
            {"seed", "", false},
            {"subsampling", "", false},
            {"shuffle", "", false},
            {"opt", "const-step,rmsprop,adagrad,adam", true},
            {"step-size", "", true},
            {"clip", "", false},
            {"decay", "", false},
            {"momentum", "", false},
            {"beta1", "", false},
            {"beta2", "", false},
            {"random-state", "", false}
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
    std::string line;
    std::getline(param_ifs, line);
    layer = std::stoi(line);
    param = seq2seq::make_tensor_tree(layer);
    tensor_tree::load_tensor(param, param_ifs);
    param_ifs.close();

    step_size = std::stod(args.at("step-size"));

    dropout = 0;
    if (ebt::in(std::string("dropout"), args)) {
        dropout = std::stod(args.at("dropout"));
    }

    if (ebt::in(std::string("clip"), args)) {
        clip = std::stod(args.at("clip"));
    }

    id_label = speech::load_label_set(args.at("label"));
    for (int i = 0; i < id_label.size(); ++i) {
        label_id[id_label[i]] = i;
    }

    seed = 1;
    if (ebt::in(std::string("seed"), args)) {
        seed = std::stod(args.at("seed"));
    }

    gen = std::default_random_engine { seed };

    if (ebt::in("random-state"s, args)) {
        std::istringstream iss { args.at("random-state") };
        iss >> gen;
    }

    indices.resize(frame_batch.pos.size());

    for (int i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }

    if (args.at("opt") == "const-step") {
        opt = std::make_shared<tensor_tree::const_step_opt>(
            tensor_tree::const_step_opt{param, step_size});
    } else if (args.at("opt") == "const-step-momentum") {
        double momentum = std::stod(args.at("momentum"));
        opt = std::make_shared<tensor_tree::const_step_momentum_opt>(
            tensor_tree::const_step_momentum_opt{param, step_size, momentum});
    } else if (args.at("opt") == "adagrad") {
        opt = std::make_shared<tensor_tree::adagrad_opt>(
            tensor_tree::adagrad_opt{param, step_size});
    } else if (args.at("opt") == "rmsprop") {
        double decay = std::stod(args.at("decay"));
        opt = std::make_shared<tensor_tree::rmsprop_opt>(
            tensor_tree::rmsprop_opt{param, step_size, decay});
    } else if (args.at("opt") == "adam") {
        double beta1 = std::stod(args.at("beta1"));
        double beta2 = std::stod(args.at("beta2"));
        opt = std::make_shared<tensor_tree::adam_opt>(
            tensor_tree::adam_opt{param, step_size, beta1, beta2});
    }

    std::ifstream opt_data_ifs { args.at("opt-data") };
    opt->load_opt_data(opt_data_ifs);
    opt_data_ifs.close();
}

void learning_env::run()
{
    ebt::Timer timer;

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

    int nsample = 0;

    while (nsample < frame_batch.pos.size()) {

        std::vector<std::vector<double>> frames = speech::load_frame_batch(frame_batch.at(nsample));
        std::vector<std::string> label_seq = speech::load_label_seq_batch(label_batch.at(nsample));

        std::vector<int> label_id_seq;
        for (auto& s: label_seq) {
            label_id_seq.push_back(label_id.at(s));
        }

        std::cout << "sample: " << nsample + 1 << std::endl;
        std::cout << "gold len: " << label_seq.size() << std::endl;

        autodiff::computation_graph comp_graph;
        std::shared_ptr<tensor_tree::vertex> var_tree
            = tensor_tree::make_var_tree(comp_graph, param);

        std::vector<double> frame_cat;
        frame_cat.reserve(frames.size() * frames.front().size());

        for (int i = 0; i < frames.size(); ++i) {
            frame_cat.insert(frame_cat.end(), frames[i].begin(), frames[i].end());
        }

        unsigned int nframes = frames.size();
        unsigned int ndim = frames.front().size();

        std::shared_ptr<autodiff::op_t> input
            = comp_graph.var(la::cpu::weak_tensor<double>(
                frame_cat.data(), { nframes, ndim }));

        input->grad_needed = false;

        std::cout << "random: " << gen << std::endl;

        std::shared_ptr<lstm::transcriber> trans;

        if (ebt::in(std::string("subsampling"), args)) {
            trans = lstm_frame::make_transcriber(param->children[0]->children[0], dropout, &gen, true);
        } else {
            trans = lstm_frame::make_transcriber(param->children[0]->children[0], dropout, &gen, false);
        }

        lstm::trans_seq_t input_seq;
        input_seq.nframes = frames.size();
        input_seq.batch_size = 1;
        input_seq.dim = frames.front().size();
        input_seq.feat = input;
        input_seq.mask = nullptr;

        lstm::trans_seq_t feat_seq = (*trans)(var_tree->children[0]->children[0], input_seq);

        std::cout << "frames: " << frames.size() << " downsampled: " << feat_seq.nframes << std::endl;

        auto decoder_nn = seq2seq::make_training_nn(label_id_seq, id_label.size(),
            feat_seq.feat, feat_seq.nframes, feat_seq.dim, var_tree);

        la::cpu::tensor<double> gold_t;
        gold_t.resize({ (unsigned int) label_id_seq.size(), (unsigned int) label_id.size() });

        for (int i = 0; i < label_id_seq.size(); ++i) {
            gold_t({i, label_id_seq[i]}) = 1;
        }

        auto& pred_t = autodiff::get_output<la::cpu::tensor_like<double>>(decoder_nn.pred);

        nn::log_loss loss { gold_t, pred_t };

        double ell = loss.loss();

        std::cout << "loss: " << ell << std::endl;
        std::cout << "E: " << ell / label_seq.size() << std::endl;

        std::shared_ptr<tensor_tree::vertex> param_grad = seq2seq::make_tensor_tree(layer);

        if (ell > 0) {
            if (decoder_nn.pred->grad == nullptr) {
                decoder_nn.pred->grad = std::make_shared<la::cpu::tensor<double>>(loss.grad());
            } else {
                auto& pred_t = autodiff::get_grad<la::cpu::tensor_like<double>>(decoder_nn.pred);
                la::cpu::copy(pred_t, loss.grad());
            }

            auto topo_order = autodiff::natural_topo_order(comp_graph);
            autodiff::guarded_grad(topo_order, autodiff::grad_funcs);
            tensor_tree::copy_grad(param_grad, var_tree);

            {
                auto vars = tensor_tree::leaves_pre_order(param_grad);

                std::cout << vars[0]->name << " "
                    << "grad norm: " << la::cpu::norm(tensor_tree::get_tensor(vars[0]))
                    << std::endl;
            }

            std::vector<std::shared_ptr<tensor_tree::vertex>> vars = tensor_tree::leaves_pre_order(param);

            double v1 = tensor_tree::get_tensor(vars[0]).data()[0];

            double grad_norm = tensor_tree::norm(param_grad);
            std::cout << "grad norm: " << grad_norm << std::endl;

            if (ebt::in(std::string("clip"), args)) {
                if (grad_norm > clip) {
                    tensor_tree::imul(param_grad, clip / grad_norm);

                    std::cout << "clip: " << clip << " gradient clipped" << std::endl;
                }
            }

            opt->update(param_grad);

            double v2 = tensor_tree::get_tensor(vars[0]).data()[0];

            std::cout << vars[0]->name << " weight: " << v1 << " update: " << v2 - v1
                << " ratio: " << (v2 - v1) / v1 << std::endl;

        }

        if (ell < 0) {
            std::cout << "loss is less than zero.  skipping." << std::endl;
        }

        double n = tensor_tree::norm(param);

        std::cout << "norm: " << n << std::endl;

        std::cout << std::endl;

        ++nsample;

#if DEBUG_TOP
        if (nsample == DEBUG_TOP) {
            break;
        }
#endif

    }

    std::ofstream param_ofs { args.at("output-param") };
    param_ofs << layer << std::endl;
    tensor_tree::save_tensor(param, param_ofs);
    param_ofs.close();

    std::ofstream opt_data_ofs { args.at("output-opt-data") };
    opt->save_opt_data(opt_data_ofs);
    opt_data_ofs.close();

}

