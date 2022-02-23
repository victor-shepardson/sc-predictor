ServerOptions.devices;

(
Server.default.options.inDevice_("Built-in Microph");
Server.default.options.outDevice_("Built-in Output");
s.options.sampleRate = 48000;
s.options.blockSize = 64;
s.boot
)

s.reboot

~bus = Bus.new(index:4, numChannels:4)

(
~synth = {
    var freq = \freq.kr(100, 0.1);
    var lr = \lr.kr(3e-3);
    var reg = \reg.kr(1e-5);
    var halluc = \halluc.kr(0, 0.1);
    /*var src = (Mix.ar(SinOsc.ar([freq, freq+10, freq*2+1, freq*5-3]))
        + (WhiteNoise.ar*1e-2)
        + (SoundIn.ar(0)*4)
        )/10;*/
    // var src = SoundIn.ar(0);
    var src = Saw.ar(freq);
    // var src = [SinOsc.ar(freq), SinOsc.ar(freq*2+1)].mean;
    var pred, resid, spars, idx;
    # pred, resid, spars, idx = Predictor.ar(src, K2A.ar(lr), K2A.ar(reg), K2A.ar(halluc));

    spars.poll(5,\sparsity);
    (SampleRate.ir/idx).poll(5,\freq);


    // Limiter.ar([Delay1.ar(src), pred])
    // Limiter.ar([Delay1.ar(src)-src, resid]) //error vs baseline
    Limiter.ar([pred, resid, LeakDC.ar(spars), idx/512]);

    // Pan2.ar(err)
    // Pan2.ar(pred)

}.play(outbus:~bus);
~bus.scope.window.bounds = Rect(0,1050,740,550);
)

48000/512

~synth.set(\freq, 352)
~synth.set(\lr,3e-2)
~synth.set(\reg, 1e-5)
~synth.set(\halluc, 0)

(
~mon.free;
~mon = Monitor.new;
~mon.play(5,1,0,2)
)

~synth.free

s.quit

s.scope