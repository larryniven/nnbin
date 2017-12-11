CXXFLAGS += -std=c++11 -I .. -L ../ebt -L ../la -L ../opt -L ../autodiff -L ../nn -L ../util
NVCCFLAGS += -std=c++11 -I .. -L ../ebt -L ../la -L ../opt -L ../autodiff -L ../nn -L ../util

.PHONY: all clean

bin = \
    fc-learn \
    fc-predict \
    fc-autoenc \
    fc-recon \
    fc-vae \
    fc-vae-recon \
    frame-tdnn-learn \
    frame-tdnn-predict \
    frame-lstm-learn \
    frame-lstm-predict \
    seq2seq-learn \
    seq2seq-predict \
    frame-win-cnn-learn \
    frame-win-cnn-predict \
    utt-autoenc-patch-learn \
    utt-autoenc-patch-recon \
    utt-autoenc-lstm-learn \
    utt-autoenc-lstm-recon \
    enhance-lstm-learn \
    enhance-lstm-predict \
    enhance-tdnn-predict

    # frame-cnn-learn \
    # frame-cnn-predict \
    # seg-lstm-learn \
    # seg-lstm-predict \
    # frame-lstm-learn-batch-gpu \
    # frame-lstm-learn-batch \
    # frame-lstm-res-learn \
    # frame-lstm-res-predict \
    # frame-lin-learn \
    # frame-lin-predict \
    # loss-lstm \
    # frame-hypercolumn-learn \
    # frame-hypercolumn-predict \
    # frame-pyramid-learn \
    # frame-pyramid-predict \
    # learn-gru \
    # predict-gru \
    # learn-residual \
    # predict-residual \
    # lstm-seg-ld-learn \
    # lstm-seg-ld-predict \
    # lstm-seg-li-learn \
    # lstm-seg-li-predict \
    # lstm-seg-li-avg \
    # lstm-seg-li-grad \
    # lstm-seg-li-update \
    # lstm-seg-logp-learn \
    # lstm-seg-logp-predict \
    # rhn-learn \
    # rhn-predict \
    # rsg-learn \
    # rsg-predict \
    # rsg-loss

gpubin = \
    frame-lstm-learn-gpu \
    frame-tdnn-learn-gpu \
    enhance-tdnn-learn-gpu

all: $(bin)

gpu: $(gpubin)

clean:
	-rm *.o
	-rm $(bin) $(gpubin)

fc-learn: fc-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

fc-predict: fc-predict.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

fc-autoenc: fc-autoenc.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

fc-recon: fc-recon.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

fc-vae: fc-vae.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

fc-vae-recon: fc-vae-recon.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

frame-tdnn-learn: frame-tdnn-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

frame-tdnn-learn-gpu.o: frame-tdnn-learn-gpu.cu
	nvcc $(NVCCFLAGS) -c frame-tdnn-learn-gpu.cu

frame-tdnn-learn-gpu: frame-tdnn-learn-gpu.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnngpu -lautodiffgpu -lutil -loptgpu -llagpu -lebt -lblas -lcublas -lcudart

frame-tdnn-predict: frame-tdnn-predict.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

frame-lstm-learn: frame-lstm-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

frame-lstm-predict: frame-lstm-predict.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

frame-lstm-learn-batch: frame-lstm-learn-batch.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

frame-lstm-learn-gpu.o: frame-lstm-learn-gpu.cu
	nvcc $(NVCCFLAGS) -c frame-lstm-learn-gpu.cu

frame-lstm-learn-gpu: frame-lstm-learn-gpu.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnngpu -lautodiffgpu -lutil -loptgpu -llagpu -lebt -lblas -lcublas -lcudart

frame-lstm-learn-batch-gpu.o: frame-lstm-learn-batch-gpu.cu
	nvcc $(NVCCFLAGS) -c frame-lstm-learn-batch-gpu.cu

frame-lstm-learn-batch-gpu: frame-lstm-learn-batch-gpu.o
	$(CXX) $(CXXFLAGS) -L /opt/cuda/lib64 -o $@ $^ -lnngpu -lautodiffgpu -loptgpu -llagpu -lutil -lebt -lblas -lcublas -lcuda -lcudart

frame-lstm-res-learn: frame-lstm-res-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

frame-lstm-res-predict: frame-lstm-res-predict.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

frame-lin-learn: frame-lin-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

frame-lin-predict: frame-lin-predict.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

loss-lstm: loss-lstm.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

frame-cnn-learn: frame-cnn-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

frame-cnn-predict: frame-cnn-predict.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

frame-win-cnn-learn: frame-win-cnn-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

frame-win-cnn-predict: frame-win-cnn-predict.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

frame-win-fc-autoenc-learn: frame-win-fc-autoenc-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

frame-win-fc-autoenc-recon: frame-win-fc-autoenc-recon.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

frame-hypercolumn-learn: frame-hypercolumn-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

frame-hypercolumn-predict: frame-hypercolumn-predict.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

frame-pyramid-learn: frame-pyramid-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

frame-pyramid-predict: frame-pyramid-predict.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

learn-gru: learn-gru.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

predict-gru: predict-gru.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

learn-residual: learn-residual.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

predict-residual: predict-residual.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

lstm-seg-ld-learn: lstm-seg-ld-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

lstm-seg-ld-predict: lstm-seg-ld-predict.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

lstm-seg-li-learn: lstm-seg-li-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

lstm-seg-li-predict: lstm-seg-li-predict.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

lstm-seg-li-avg: lstm-seg-li-avg.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

lstm-seg-li-grad: lstm-seg-li-grad.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

lstm-seg-li-update: lstm-seg-li-update.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

lstm-seg-logp-learn: lstm-seg-logp-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

lstm-seg-logp-predict: lstm-seg-logp-predict.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

rhn-learn: rhn-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

rhn-predict: rhn-predict.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

rsg-learn: rsg-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

rsg-predict: rsg-predict.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

rsg-loss: rsg-loss.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

seg-lstm-learn: seg-lstm-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

seg-lstm-predict: seg-lstm-predict.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

seq2seq-learn: seq2seq-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

seq2seq-predict: seq2seq-predict.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

utt-autoenc-patch-learn: utt-autoenc-patch-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

utt-autoenc-patch-recon: utt-autoenc-patch-recon.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

utt-autoenc-lstm-learn: utt-autoenc-lstm-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

utt-autoenc-lstm-recon: utt-autoenc-lstm-recon.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

enhance-lstm-learn: enhance-lstm-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

enhance-lstm-predict: enhance-lstm-predict.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

enhance-tdnn-learn-gpu.o: enhance-tdnn-learn-gpu.cu
	nvcc $(NVCCFLAGS) -c enhance-tdnn-learn-gpu.cu

enhance-tdnn-learn-gpu: enhance-tdnn-learn-gpu.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnngpu -lautodiffgpu -lutil -loptgpu -llagpu -lebt -lblas -lcublas -lcudart

enhance-tdnn-predict: enhance-tdnn-predict.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lutil -lopt -lla -lebt -lblas

