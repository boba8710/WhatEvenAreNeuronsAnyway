import java.util.Random;
import java.lang.Math;
public class Neuron {
	//private double[] inputs; //Storing inputs in each neuron is just not supportable. Way too much memory used. Needs to get needed
							 //input values at runtime. THIS MUST BE FIXED.
	private double[] weights;
	private boolean outputComputedForSettings = false;
	private double output;
	private double neuronBias;
	
	private boolean DEBUG = false;
	
	public Neuron(int inputCount, double neuronBias){
		if(DEBUG){
			System.out.println("[d] Initialized a full neuron");
		}
		this.weights = new double[inputCount];
		this.neuronBias = neuronBias;
	}
	public double getNeuronBias() {
		return neuronBias;
	}
	public Neuron(double neuronBias){
		if(DEBUG){
			System.out.println("[d] Initialized an empty neuron");
		}
		this.neuronBias = neuronBias;
	}
	public void setInputCount(int inputCount){
		if(DEBUG){
			System.out.println("[d] Input count set");
		}
		this.weights = new double[inputCount];
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
	public int getWeightCount(){
		return weights.length;
	}
	public double[] getWeights(){
		return this.weights;
	}
}
