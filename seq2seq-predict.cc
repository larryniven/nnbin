#include "util/speech.h"
#include "util/batch.h"
#include "util/util.h"
#include <fstream>
#include "ebt/ebt.h"
#include "nn/lstm-frame.h"
#include "nn/seq2seq.h"

struct prediction_env {

    std::ifstream frame_batch;

    int layer;
    std::shared_ptr<tensor_tree::vertex> param;

    std::unordered_map<std::string, int> label_id;
    std::vector<std::string> id_label;

    std::unordered_map<std::string, std::string> args;

    prediction_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "seq2seq-predict",
        "Find 1-best path",
        {
            {"frame-batch", "", false},
            {"param", "", true},
            {"label", "", true},
            {"attention", "", true},
            {"subsampling", "", false},
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

    prediction_env env { args };

    env.run();

    return 0;
}

prediction_env::prediction_env(std::unordered_map<std::string, std::string> args)
    : args(args)
{
    frame_batch.open(args.at("frame-batch"));

    std::ifstream param_ifs { args.at("param") };
    std::string line;
    std::getline(param_ifs, line);
    layer = std::stoi(line);
    param = seq2seq::make_tensor_tree(layer);
    tensor_tree::load_tensor(param, param_ifs);
    param_ifs.close();

    id_label = util::load_label_set(args.at("label"));
    for (int i = 0; i < id_label.size(); ++i) {
        label_id[id_label[i]] = i;
    }
}

void prediction_env::run()
{
    ebt::Timer timer;

    int nsample = 0;

    while (1) {

        std::vector<std::vector<double>> frames = speech::load_frame_batch(frame_batch);

        if (!frame_batch) {
            break;
        }

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

        std::shared_ptr<lstm::transcriber> trans;

        if (ebt::in(std::string("subsampling"), args)) {
            trans = lstm_frame::make_transcriber(param->children[0]->children[0], 0.0, nullptr, true);
        } else {
            trans = lstm_frame::make_transcriber(param->children[0]->children[0], 0.0, nullptr, false);
        }

        lstm::trans_seq_t input_seq;
        input_seq.nframes = frames.size();
        input_seq.batch_size = 1;
        input_seq.dim = frames.front().size();
        input_seq.feat = input;
        input_seq.mask = nullptr;

        lstm::trans_seq_t feat_seq = (*trans)(var_tree->children[0]->children[0], input_seq);

        std::shared_ptr<seq2seq::attention> att_func;

        if (args.at("attention") == "bilin") {
            att_func = std::make_shared<seq2seq::bilinear_attention>(
                seq2seq::bilinear_attention{});
        } else if (args.at("attention") == "bilin-softmax") {
            att_func = std::make_shared<seq2seq::bilinear_softmax_attention>(
                seq2seq::bilinear_softmax_attention{});
        } else {
            throw std::logic_error("unknown attention " + args.at("attention"));
        }

        std::vector<int> labels = seq2seq::decode(id_label,
            feat_seq.feat, feat_seq.nframes, feat_seq.dim, var_tree, *att_func);

        for (int i = 0; i < labels.size() - 1; ++i) {
            std::cout << id_label[labels[i]] << " ";
        }

        std::cout << "(" << nsample << ".dot)" << std::endl;

#if DEBUG_TOP
        if (nsample == DEBUG_TOP) {
            break;
        }
#endif

        ++nsample;

    }

}

