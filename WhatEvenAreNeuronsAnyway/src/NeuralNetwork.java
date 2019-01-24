import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
public class NeuralNetwork {
	private ArrayList<Neuron> inputLayer = new ArrayList<Neuron>();
	private ArrayList<ArrayList<Neuron>> hiddenLayers = new ArrayList<ArrayList<Neuron>>();
	private ArrayList<Neuron> outputLayer = new ArrayList<Neuron>();
	
	private boolean DEBUG = false;
	
	private boolean inputWebCreated=false;
	
	private double networkScore = 0;
	public NeuralNetwork(int inputLayerNeuronCount, int outputLayerNeuronCount){
		if(DEBUG){
			System.out.println("[d] Initialized new neural network");
		}
		for(int i = 0; i < inputLayerNeuronCount; i++){
			Neuron n = new Neuron(0);
			inputLayer.add(n);
		}
		if(DEBUG){
			System.out.println("[d] Input Neuron initialization complete");
		}
		for(int i = 0; i < outputLayerNeuronCount; i++){
			Neuron n = new Neuron(0);
			outputLayer.add(n);
		}
		if(DEBUG){
			System.out.println("[d] Output Neuron initialization complete");
		}
	}
	/**
	 * To be used after all hidden layers are added, makes sure each layer has as many
	 * inputs as there are neurons in the last layer. Throws and InputWebException if the input web has already been
	 * created when the function is called.
	 * @throws InputWebException
	 */
	public void createInputWeb() throws InputWebException{
		if(DEBUG){
			System.out.println("[d] Started creation of input web");
		}
		if(inputWebCreated==true){
			throw new InputWebException();
		}
		inputWebCreated=true;
		int inputLayerNeuronCount = inputLayer.size();
		if(DEBUG){
			System.out.println("[d] Starting configuration of hidden layer inputs");
		}
		for(int i = 0; i<hiddenLayers.size();i++){
			if(i==0){
				for(Neuron n : hiddenLayers.get(i)){
					n.setInputCount(inputLayerNeuronCount);
				}
			}else{
				for(Neuron n : hiddenLayers.get(i)){
					n.setInputCount(hiddenLayers.get(i-1).size());
				}
			}
		}
		//Set the number of inputs for the output layer neurons to the number of neurons in the last hidden layer
		for(Neuron n : outputLayer){
			n.setInputCount(hiddenLayers.get(hiddenLayers.size()-1).size()); 
			for(int i = 0; i < n.getWeightCount(); i++){
				n.getWeights()[i] = 1;
			}
		}
		if(DEBUG){
			System.out.println("[d] Finished creation of input web");
		}
	}
	/**
	 * This function adds a new hidden layer with neuronCount neurons and throws
	 * an InputWebException if attempting to add another layer to a network with an already-constructed
	 * input web.
	 * @param neuronCount
	 * @throws InputWebException
	 */
	public void addHiddenLayer(int neuronCount) throws InputWebException{
		if(DEBUG){
			System.out.println("[d] Starting add of hidden layer");
		}
		if(inputWebCreated==true){
			throw new InputWebException();
		}
		ArrayList<Neuron> hiddenLayer = new ArrayList<Neuron>();
		hiddenLayers.add(hiddenLayer);
		for(int i = 0; i < neuronCount; i++){
			Neuron n = new Neuron(0);
			hiddenLayers.get(hiddenLayers.size()-1).add(n);
		}
		if(DEBUG){
			System.out.println("[d] Hidden layer added");
		}
	}
	
	private void computeNeuronOutput(Neuron neuron, ArrayList<Double> previousLayerScores){
			double neuronScore = neuron.getNeuronBias();
			for(int k = 0; k < previousLayerScores.size(); k++){
				neuronScore+=previousLayerScores.get(k)*neuron.getSingleWeight(k);
			}
			neuron.overrideOutput(neuronScore);
	}
	public void computeOutput(){ //MUST be multithreaded for use in networks of any useful size
		ArrayList<Double> previousLayerScores = new ArrayList<Double>();
		for(Neuron n : inputLayer){
			previousLayerScores.add(n.getOutput());
		}
		for(int i = 0; i < hiddenLayers.size();i++){
			
			ArrayList<Neuron> hiddenLayer = hiddenLayers.get(i);
			//This should be where the multithreading happens. Add this to a multithreaded method for fun and speedy profit
			final ExecutorService executor = Executors.newCachedThreadPool();
			final List<Future<?>> futures = new ArrayList<>();
			for(Neuron neuron : hiddenLayer) {
				Future<?> future = executor.submit(() -> {
					computeNeuronOutput(neuron,previousLayerScores);
				});
				futures.add(future);
			}
			try {
		        for (Future<?> future : futures) {
		            future.get();
		        }
		    } catch (Exception e) {
		        e.printStackTrace();
		    }
			
			
			
			previousLayerScores.clear();
			for(Neuron n : hiddenLayers.get(i)){
				previousLayerScores.add(n.getOutput());
			}
		}
		for(int i = 0; i < outputLayer.size();i++){
			double neuronScore = outputLayer.get(i).getNeuronBias();
			for(int j = 0; j < previousLayerScores.size(); j++){
				neuronScore+=previousLayerScores.get(j)*outputLayer.get(i).getSingleWeight(j);
			}
			outputLayer.get(i).overrideOutput(neuronScore);
		}
	}
	/**
	 * Randomizes all the weights in this network's hidden layers (-1<x<1)
	 */
	public void randomizeAllHiddenLayerWeights(){
		if(DEBUG){
			System.out.println("[d] Randomizing hidden layer weights");
		}
		for(ArrayList<Neuron> hiddenLayer : hiddenLayers){
			for(Neuron neuron : hiddenLayer){
				neuron.setRandomWeights();
			}
		}
		if(DEBUG){
			System.out.println("[d] Randomization completed");
		}
	}
	/**
	 * Randomizes all the biases in this network's hidden layers (-100<x<100)
	 */
	public void randomizeAllHiddenLayerBiases(){
		if(DEBUG){
			System.out.println("[d] Randomizing hidden layer biases");
		}
		for(ArrayList<Neuron> hiddenLayer : hiddenLayers){
			for(Neuron neuron : hiddenLayer){
				neuron.setRandomBias();
			}
		}
		if(DEBUG){
			System.out.println("[d] Randomization completed");
		}
	}
	/**
	 * Takes an array of the same size as the input layer and stores it's values into the input layer
	 */
	public void initializeInputLayer(double[] input){
		if(DEBUG){
			System.out.println("[d] Overriding outputs for input layer neurons");
		}
		assert input.length==inputLayer.size();
		for(int i = 0; i < inputLayer.size(); i++){
			inputLayer.get(i).overrideOutput(input[i]);
		}
		if(DEBUG){
			System.out.println("[d] Override completed");
		}
	}
	@Deprecated
	public void computeOutputV1(){
		/*for(Neuron hiddenNeuron : hiddenLayers.get(0)){
			double[] inputLayerOutputs = new double[inputLayer.size()];
			for(int i = 0; i<inputLayer.size();i++){
				inputLayerOutputs[i]=inputLayer.get(i).getOutput();
			}
			hiddenNeuron.readFromInputs(inputLayerOutputs);
			hiddenNeuron.computeOutput();
		}
		for(int i = 1; i < hiddenLayers.size(); i++){
			double[] previousLayerOutputs = new double[hiddenLayers.get(i-1).size()];
			for(int j = 0; j<hiddenLayers.get(i-1).size();j++){
				previousLayerOutputs[j]=hiddenLayers.get(i-1).get(j).getOutput();
			}
			
			for(Neuron hiddenNeuron : hiddenLayers.get(i)){
				hiddenNeuron.readFromInputs(previousLayerOutputs);
				
				hiddenNeuron.computeOutput();
				
			}
		}
		for(Neuron outputNeuron : outputLayer){
			double[] previousLayerOutputs = new double[hiddenLayers.get(hiddenLayers.size()-1).size()];
			for(int j = 0; j<hiddenLayers.get(hiddenLayers.size()-1).size();j++){
				previousLayerOutputs[j]=hiddenLayers.get(hiddenLayers.size()-1).get(j).getOutput();
			}
			outputNeuron.readFromInputs(previousLayerOutputs);
			outputNeuron.computeOutput();
		}*/
	}
	public ArrayList<Neuron> getOutputLayer(){
		return this.outputLayer;
	}
	public double getNetworkScore() {
		return networkScore;
	}
	public void setNetworkScore(double networkScore) {
		this.networkScore = networkScore;
	}
	public ArrayList<ArrayList<Neuron>> getHiddenLayers(){
		return this.hiddenLayers;
	}
	public void setNeuron(int hiddenLayerIndex, int neuronIndex, Neuron neuron){
		 hiddenLayers.get(hiddenLayerIndex).set(neuronIndex, neuron);
	}
	public NeuralNetwork deepCopy() throws InputWebException{
		NeuralNetwork clonedNetwork = new NeuralNetwork(this.inputLayer.size(), this.outputLayer.size());
		for(int i = 0; i < this.hiddenLayers.size(); i++){
			clonedNetwork.addHiddenLayer(this.hiddenLayers.get(i).size());
		}
		clonedNetwork.createInputWeb();
		for(int i = 0; i < this.hiddenLayers.size(); i++){
			for(int j = 0; j < this.hiddenLayers.get(i).size(); j++){
				clonedNetwork.hiddenLayers.get(i).get(j).setBias(this.hiddenLayers.get(i).get(j).getNeuronBias());
				for(int k = 0; k < this.hiddenLayers.get(i).get(j).getWeightCount(); k++){
					clonedNetwork.hiddenLayers.get(i).get(j).setSingleWeight(k, this.hiddenLayers.get(i).get(j).getSingleWeight(k));
				}
			}
		}
		return clonedNetwork;
	}
	public void copyFromNetwork(NeuralNetwork input){
		assert input.hiddenLayers.size() == this.hiddenLayers.size();
		for(int i = 0; i < input.hiddenLayers.size(); i++){
			for(int j = 0; j < input.hiddenLayers.get(i).size(); j++){
				this.hiddenLayers.get(i).get(j).setBias(input.hiddenLayers.get(i).get(j).getNeuronBias());
				for(int k = 0; k < input.hiddenLayers.get(i).get(j).getWeightCount(); i++){
					this.hiddenLayers.get(i).get(j).setSingleWeight(k, input.hiddenLayers.get(i).get(j).getSingleWeight(k));
				}
			}
		}
	}
}
