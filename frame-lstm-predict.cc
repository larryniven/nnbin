#include "ebt/ebt.h"
#include "util/speech.h"
#include "util/batch.h"
#include "util/util.h"
#include "nn/lstm.h"
#include "nn/tensor-tree.h"
#include "nn/lstm-frame.h"
#include <fstream>

struct prediction_env {

    batch::scp frame_scp;

    int layer;
    std::shared_ptr<tensor_tree::vertex> param;

    std::vector<std::string> label;

    std::unordered_map<std::string, std::string> args;

    prediction_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "frame-lstm-predict",
        "Predict frames with LSTM",
        {
            {"frame-scp", "", true},
            {"param", "", true},
            {"label", "", true},
            {"layer", "", false},
            {"print-logprob", "", false},
            {"print-hidden", "", false},
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
    frame_scp.open(args.at("frame-scp"));

    std::ifstream param_ifs { args.at("param") };
    std::string line;
    std::getline(param_ifs, line);
    layer = std::stoi(line);
    param = lstm_frame::make_tensor_tree(layer);
    tensor_tree::load_tensor(param, param_ifs);
    param_ifs.close();

    label = util::load_label_set(args.at("label"));

    if (ebt::in(std::string("layer"), args)) {
        layer = std::stoi(args.at("layer"));
    }
}

void prediction_env::run()
{
    int nsample = 0;

    while (nsample < frame_scp.entries.size()) {
        std::vector<std::vector<double>> frames = speech::load_frame_batch(frame_scp.at(nsample));

        autodiff::computation_graph graph;
        std::vector<double> input_vec;
        input_vec.reserve(frames.size() * frames.front().size());

        for (int i = 0; i < frames.size(); ++i) {
            input_vec.insert(input_vec.end(), frames[i].begin(), frames[i].end());
        }

        std::shared_ptr<autodiff::op_t> input = graph.var(
            la::cpu::weak_tensor<double>(input_vec.data(),
            {(unsigned int) frames.size(), (unsigned int) frames.front().size()}));

        std::shared_ptr<tensor_tree::vertex> var_tree = tensor_tree::make_var_tree(graph, param);

        std::shared_ptr<lstm::transcriber> trans;

        trans = lstm_frame::make_transcriber(param->children[0], 0.0, nullptr, false);

        lstm::trans_seq_t input_seq;
        input_seq.nframes = frames.size();
        input_seq.batch_size = 1;
        input_seq.dim = frames.front().size();
        input_seq.feat = input;
        input_seq.mask = nullptr;

        lstm::trans_seq_t feat_seq = (*trans)(var_tree->children[0], input_seq);

        lstm::trans_seq_t output_seq;

        if (ebt::in(std::string("print-hidden"), args)) {
            output_seq = feat_seq;
        } else {
            lstm::fc_transcriber fc_trans { (int) label.size() };
            lstm::logsoftmax_transcriber logsoftmax_trans;
            auto score = fc_trans(var_tree->children[1], feat_seq);

            output_seq = logsoftmax_trans(nullptr, score);
        }

        std::shared_ptr<autodiff::op_t> output = output_seq.feat;

        std::cout << nsample << ".phn" << std::endl;

        auto& output_t = autodiff::get_output<la::cpu::tensor_like<double>>(output);

        if (ebt::in(std::string("print-logprob"), args) || ebt::in(std::string("print-hidden"), args)) {
            for (int t = 0; t < output_t.size(0); ++t) {
                std::cout << output_t({t, 0, 0});

                for (int j = 1; j < output_t.size(2); ++j) {
                    std::cout << " " << output_t({t, 0, j});
                }

                std::cout << std::endl;
            }
        } else {
            for (int t = 0; t < output_t.size(0); ++t) {
                int argmax = -1;
                double max = -std::numeric_limits<double>::infinity();

                for (int j = 0; j < output_t.size(2); ++j) {
                    if (output_t({t, 0, j}) > max) {
                        max = output_t({t, 0, j});
                        argmax = j;
                    }
                }

                if (t != 0) {
                    std::cout << " ";
                }

                std::cout << label[argmax];
            }

            std::cout << std::endl;
        }

        std::cout << "." << std::endl;

        ++nsample;
    }
}

