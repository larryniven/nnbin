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

struct prediction_env {

    std::ifstream frame_batch;
    std::ifstream seg_batch;

    int layer;
    std::shared_ptr<tensor_tree::vertex> param;

    int output_dim;

    std::default_random_engine gen;

    std::vector<std::string> id_label;
    std::unordered_map<std::string, int> label_id;

    std::unordered_map<std::string, std::string> args;

    prediction_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "seg-lstm-predict",
        "Decode with a LSTM segment classifier",
        {
            {"frame-batch", "", true},
            {"seg-batch", "", true},
            {"param", "", true},
            {"label", "", true},
            {"seg-lstm", "", false},
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
    seg_batch.open(args.at("seg-batch"));

    std::string line;

    std::ifstream param_ifs { args.at("param") };
    std::getline(param_ifs, line);
    layer = std::stoi(line);
    param = make_tensor_tree(layer);
    tensor_tree::load_tensor(param, param_ifs);
    param_ifs.close();

    output_dim = tensor_tree::get_tensor(param->children[1]->children[0]).size(1);

    id_label = speech::load_label_set(args.at("label"));
    for (int i = 0; i < id_label.size(); ++i) {
        label_id[id_label[i]] = i;
    }
}

void prediction_env::run()
{
    int nsample = 0;

    while (true) {
        std::vector<std::vector<double>> frames = speech::load_frame_batch(frame_batch);
	std::vector<speech::segment> segs = speech::load_segment_batch(seg_batch);

        if (!frame_batch || !seg_batch) {
            break;
        }

        autodiff::computation_graph graph;

        std::vector<double> input_cat;
        input_cat.reserve(frames.size() * frames.front().size());

        for (int i = 0; i < frames.size(); ++i) {
            input_cat.insert(input_cat.end(), frames[i].begin(), frames[i].end());
        }

        std::shared_ptr<autodiff::op_t> input = graph.var(
            la::cpu::tensor<double>(la::cpu::weak_vector<double>(input_cat.data(), input_cat.size()),
            { (unsigned int) frames.size(), (unsigned int) frames.front().size() }));

        std::shared_ptr<tensor_tree::vertex> var_tree = tensor_tree::make_var_tree(graph, param);

        lstm::trans_seq_t input_seq;
        input_seq.nframes = frames.size();
        input_seq.batch_size = 1;
        input_seq.dim = frames.front().size();
        input_seq.feat = input;
        input_seq.mask = nullptr;

        std::shared_ptr<lstm::transcriber> trans;
        trans = lstm_frame::make_transcriber(param->children[0], 0.0, nullptr, false);
        lstm::trans_seq_t frame_feat = (*trans)(var_tree->children[0], input_seq);

        auto seg_feat = transcribe_seg_feat(var_tree->children[1], frame_feat, segs, output_dim);

        if (ebt::in(std::string("seg-lstm"), args)) {
            trans = lstm_frame::make_transcriber(param->children[2], 0.0, nullptr, false);
            seg_feat = (*trans)(var_tree->children[2], seg_feat);
        }

        lstm::fc_transcriber fc_trans { (int) label_id.size() };
        lstm::logsoftmax_transcriber logsoftmax_trans;
        auto score_seq = fc_trans(var_tree->children[3], seg_feat);
        auto output_seq = logsoftmax_trans(nullptr, score_seq);

        std::shared_ptr<autodiff::op_t> logprob = output_seq.feat;
        auto& logprob_t = autodiff::get_output<la::cpu::tensor_like<double>>(logprob);

        double inf = std::numeric_limits<double>::infinity();

        std::cout << nsample << ".seg" << std::endl;

        for (int i = 0; i < segs.size(); ++i) {
            double max = -inf;
            int argmax = -1;

            for (int k = 0; k < id_label.size(); ++k) {
                if (logprob_t({i, k}) > max) {
                    max = logprob_t({i, k});
                    argmax = k;
                }
            }

            std::cout << segs[i].start_time << " " << segs[i].end_time
                << " " << id_label[argmax] << std::endl;
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

