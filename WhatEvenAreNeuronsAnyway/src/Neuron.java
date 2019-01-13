import java.util.Random;
import java.lang.Math;
public class Neuron {
	private double[] inputs;
	private double[] weights;
	private boolean outputComputedForSettings = false;
	private double output;
	private double neuronBias;
	public Neuron(int inputCount, double neuronBias){
		if(Globals.DEBUG){
			System.out.println("[d] Initialized a full neuron");
			Globals.neuronCount++;
			System.out.println("[d] Neuron count: "+Globals.neuronCount);
		}
		this.inputs  = new double[inputCount];
		this.weights = new double[inputCount];
		this.neuronBias = neuronBias;
	}
	public Neuron(double neuronBias){
		if(Globals.DEBUG){
			System.out.println("[d] Initialized an empty neuron");
			Globals.neuronCount++;
			System.out.println("[d] Neuron count: "+Globals.neuronCount);
		}
		this.neuronBias = neuronBias;
	}
	public void setInputCount(int inputCount){
		if(Globals.DEBUG){
			System.out.println("[d] Input count set");
		}
		this.inputs  = new double[inputCount];
		this.weights = new double[inputCount];
	}
	//This can be safely multithreaded as inputs from the previous layer don't change
	public void computeOutput(){
		if(Globals.DEBUG){
			System.out.println("[d] Computing neuron output");
		}
		output = neuronBias;
		for(int i = 0; i < inputs.length; i++){
			output+=inputs[i]*weights[i];
		}
		this.outputComputedForSettings=true;
		if(Globals.DEBUG){
			System.out.println("[d] Output computation completed");
		}
	}
	/**
	 * Sets the output for this neuron
	 * @param output
	 */
	public void overrideOutput(double output){
		this.output=output;
		this.outputComputedForSettings=true;
	}
	public double getOutput(){
		assert this.outputComputedForSettings==true;
		return 2/Math.PI*Math.atan(Math.PI*this.output);
	}
	/**
	 * Randomizes all weights for this neuron
	 */
	public void setRandomWeights(){
		this.outputComputedForSettings=false;
		Random r = new Random();
		for(double weight : weights){
			weight = (r.nextDouble()>0.5 ? 1:-1)*r.nextDouble();
		}
	}
	/**
	 * Randomizes bias for neurons
	 */
	public void setRandomBias(){
		this.outputComputedForSettings=false;
		Random r = new Random();
		this.neuronBias = (r.nextDouble()>0.5 ? 1:-1)*r.nextDouble();
	}
	public void setBias(double bias){
		this.outputComputedForSettings=false;
		this.neuronBias = bias;
	}
	public double getSingleWeight(int index) {
		return weights[index];
	}
	public void setSingleWeight(int index, double weight){
		this.outputComputedForSettings=false;
		weights[index]=weight;
	}
	public void randomizeSingleWeight(int index){
		this.outputComputedForSettings=false;
		Random r = new Random();
		weights[index] = (r.nextDouble()>0.5 ? 1:-1)*r.nextDouble();
	}
	public boolean hasOutputBeenComputed() {
		return outputComputedForSettings;
	}
	public void readFromInputs(double[] inputs){
		assert(this.inputs.length == inputs.length);
		for(int i = 0; i < inputs.length; i++){
			this.inputs[i]=inputs[i];
		}
	}
	public int getInputCount(){
		return inputs.length;
	}
}
