CXXFLAGS += -std=c++14 -I .. -L ../ebt -L ../la -L ../opt -L ../autodiff -L ../nn -L ../speech
NVCCFLAGS += -std=c++11 -I .. -L ../ebt -L ../la -L ../opt -L ../autodiff -L ../nn -L ../speech

.PHONY: all clean

bin = \
    frame-lstm-learn \
    frame-lstm-predict \
    frame-cnn-learn \
    frame-cnn-predict \
    seg-lstm-learn \
    seg-lstm-predict

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


all: $(bin)

clean:
	-rm *.o
	-rm $(bin)

frame-lstm-learn: frame-lstm-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

frame-lstm-predict: frame-lstm-predict.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

frame-lstm-learn-batch: frame-lstm-learn-batch.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

frame-lstm-learn-batch-gpu.o: frame-lstm-learn-batch-gpu.cu
	nvcc $(NVCCFLAGS) -c frame-lstm-learn-batch-gpu.cu

frame-lstm-learn-batch-gpu: frame-lstm-learn-batch-gpu.o
	$(CXX) $(CXXFLAGS) -L /opt/cuda/lib64 -o $@ $^ -lnngpu -lautodiffgpu -lspeech -loptgpu -llagpu -lebt -lblas -lcublas -lcuda -lcudart

frame-lstm-res-learn: frame-lstm-res-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

frame-lstm-res-predict: frame-lstm-res-predict.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

frame-lin-learn: frame-lin-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

frame-lin-predict: frame-lin-predict.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

loss-lstm: loss-lstm.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

frame-cnn-learn: frame-cnn-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

frame-cnn-predict: frame-cnn-predict.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

frame-win-cnn-learn: frame-win-cnn-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

frame-hypercolumn-learn: frame-hypercolumn-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

frame-hypercolumn-predict: frame-hypercolumn-predict.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

frame-pyramid-learn: frame-pyramid-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

frame-pyramid-predict: frame-pyramid-predict.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

learn-gru: learn-gru.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

predict-gru: predict-gru.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

learn-residual: learn-residual.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

predict-residual: predict-residual.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

lstm-seg-ld-learn: lstm-seg-ld-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

lstm-seg-ld-predict: lstm-seg-ld-predict.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

lstm-seg-li-learn: lstm-seg-li-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

lstm-seg-li-predict: lstm-seg-li-predict.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

lstm-seg-li-avg: lstm-seg-li-avg.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

lstm-seg-li-grad: lstm-seg-li-grad.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

lstm-seg-li-update: lstm-seg-li-update.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

lstm-seg-logp-learn: lstm-seg-logp-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

lstm-seg-logp-predict: lstm-seg-logp-predict.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

rhn-learn: rhn-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

rhn-predict: rhn-predict.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

rsg-learn: rsg-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

rsg-predict: rsg-predict.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

rsg-loss: rsg-loss.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

seg-lstm-learn: seg-lstm-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

seg-lstm-predict: seg-lstm-predict.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

