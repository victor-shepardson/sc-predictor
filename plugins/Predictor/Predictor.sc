Predictor : MultiOutUGen {
	*ar { |input, lr=1e-3, reg=0, halluc=0|
		^this.multiNew('audio', input, lr, reg, halluc);
	}
	checkInputs {
		/* TODO */
		^this.checkValidInputs;
	}
	init {arg ... theInputs;
		inputs = theInputs;
		^this.initOutputs(4, rate);
	}
}
