ServerOptions.devices;

(
Server.default.options.inDevice_("K-Mix");
Server.default.options.outDevice_("K-Mix");
s.options.numInputBusChannels = 8;
s.options.numOutputBusChannels = 8;
s.options.sampleRate = 48000;
s.options.blockSize = 64;
s.boot;
~outbuses = [0,1];
~inbuses = [6,7];
)

s.reboot

// ~bus = Bus.new(index:4, numChannels:4)


(
~synth = {
    // var freq = \freq.kr(100, 0.1);
    var lr = \lr.kr(3e-3);
    var l1 = \l1.kr(0);
    var l2 = \l2.kr(1e-5);
    var halluc = \halluc.kr(0, 0.1);
    /*var src = (Mix.ar(SinOsc.ar([freq, freq+10, freq*2+1, freq*5-3]))
        + (WhiteNoise.ar*1e-2)
        + (SoundIn.ar(0)*4)
        )/10;*/
    // var src = SoundIn.ar(0);
    var src = SoundIn.ar(~inbuses);
    // var src = [SinOsc.ar(freq), SinOsc.ar(freq*2+1)].mean;
    var pred, resid, spars, idx, out;
    # pred, resid, spars, idx = Predictor.ar(src,
        K2A.ar(lr), K2A.ar(l1), K2A.ar(l2), K2A.ar(halluc)).flop;

    // spars.poll(5,\sparsity);
    (SampleRate.ir/idx).poll(5,\freq);
    // pred.poll(5);
    // src.poll(5);
    // resid.poll(5);


    // out = Integrator.ar(HPF.ar(resid, 20), 0.99);
    out = HPF.ar(out, 20);
    out = Integrator.ar(resid, 0.99);
    out = FreqShift.ar(out, 200 - (SampleRate.ir/idx));
    out = HPF.ar(out, 20);

    Limiter.ar(out, 0.5, 0.03)
    // Limiter.ar([src/2,out], 0.5, 0.03)


    // Pan2.ar(err)
    // Pan2.ar(pred)

}.play(outbus:~outbuses[0]);
// ~bus.scope.window.bounds = Rect(0,1050,740,550);
)

48000/512

~synth.set(\lr,1e-1)
~synth.set(\l1, 1e-3)
~synth.set(\l2, 1e-3)
~synth.set(\halluc, 0)

(/*
~mon.free;
~mon = Monitor.new;
~mon.play(5,1,0,2)
)*/

~synth.free

s.quit

s.scope