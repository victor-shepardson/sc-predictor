Predictor : MultiOutUGen {
	*ar { |input, lr=1e-3, l1=0, l2=0, halluc=0|
		^this.multiNew('audio', input, lr, l1, l2, halluc);
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
