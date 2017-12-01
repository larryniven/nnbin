#include "la/la-cpu.h"
#include "autodiff/autodiff.h"
#include "ebt/ebt.h"
#include "util/speech.h"
#include "util/batch.h"
#include "util/util.h"
#include <fstream>
#include <vector>
#include "nn/lstm.h"
#include "nn/lstm-tensor-tree.h"
#include "nn/nn.h"
#include <random>
#include "nn/tensor-tree.h"
#include "nn/lstm-frame.h"
#include <algorithm>

struct prediction_env {

    batch::scp noisy_scp;

    int layer;
    std::shared_ptr<tensor_tree::vertex> param;

    std::vector<int> indices;

    std::unordered_map<std::string, std::string> args;

    prediction_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "enhance-lstm-predict",
        "Enhance features with an LSTM",
        {
            {"noisy-scp", "", true},
            {"param", "", true},
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
    noisy_scp.open(args.at("noisy-scp"));

    std::string line;

    std::ifstream param_ifs { args.at("param") };
    std::getline(param_ifs, line);
    layer = std::stoi(line);
    param = lstm_frame::make_tensor_tree(layer);

    tensor_tree::load_tensor(param, param_ifs);
    param_ifs.close();

    indices.resize(noisy_scp.entries.size());

    for (int i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }
}

void prediction_env::run()
{
    int nsample = 0;

    while (nsample < indices.size()) {
        std::vector<std::vector<double>> noisy_frames = speech::load_frame_batch(noisy_scp.at(indices[nsample]));

        autodiff::computation_graph graph;

        std::vector<double> input_cat;
        input_cat.reserve(noisy_frames.size() * noisy_frames.front().size());

        for (int i = 0; i < noisy_frames.size(); ++i) {
            input_cat.insert(input_cat.end(), noisy_frames[i].begin(), noisy_frames[i].end());
        }

        std::shared_ptr<autodiff::op_t> input = graph.var(
            la::cpu::tensor<double>(la::cpu::weak_vector<double>(input_cat.data(), input_cat.size()),
            { (unsigned int) noisy_frames.size(), 1, (unsigned int) noisy_frames.front().size() }));

        input->grad_needed = false;

        std::shared_ptr<tensor_tree::vertex> var_tree = tensor_tree::make_var_tree(graph, param);

        lstm::trans_seq_t input_seq;
        input_seq.nframes = noisy_frames.size();
        input_seq.batch_size = 1;
        input_seq.dim = noisy_frames.front().size();
        input_seq.feat = input;
        input_seq.mask = nullptr;

        std::shared_ptr<lstm::transcriber> trans = lstm_frame::make_transcriber(
            param->children[0], 0.0, nullptr, false);

        lstm::trans_seq_t feat_seq = (*trans)(var_tree->children[0], input_seq);

        lstm::fc_transcriber fc_trans { (int) noisy_frames.front().size() };
        auto output_seq = fc_trans(var_tree->children[1], feat_seq);

        std::shared_ptr<autodiff::op_t> output = output_seq.feat;

        auto& pred = autodiff::get_output<la::cpu::tensor_like<double>>(output);

        std::cout << noisy_scp.entries[nsample].key << std::endl;

        for (int i = 0; i < pred.size(0); ++i) {
            for (int j = 0; j < pred.size(2); ++j) {
                std::cout << pred({i, 0, j}) << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "." << std::endl;

        ++nsample;

#if DEBUG_TOP
        if (nsample >= DEBUG_TOP) {
            break;
        }
#endif
    }

}

