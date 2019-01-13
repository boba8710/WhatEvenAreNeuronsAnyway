import java.util.ArrayList;
public class NeuralNetwork {
	private ArrayList<Neuron> inputLayer = new ArrayList<Neuron>();
	private ArrayList<ArrayList<Neuron>> hiddenLayers = new ArrayList<ArrayList<Neuron>>();
	private ArrayList<Neuron> outputLayer = new ArrayList<Neuron>();
	
	private boolean inputWebCreated=false;
	
	private double networkScore = 0;
	public NeuralNetwork(int inputLayerNeuronCount, int outputLayerNeuronCount){
		if(Globals.DEBUG){
			System.out.println("[d] Initialized new neural network");
		}
		for(int i = 0; i < inputLayerNeuronCount; i++){
			Neuron n = new Neuron(0);
			inputLayer.add(n);
		}
		if(Globals.DEBUG){
			System.out.println("[d] Input Neuron initialization complete");
		}
		for(int i = 0; i < outputLayerNeuronCount; i++){
			Neuron n = new Neuron(0);
			outputLayer.add(n);
		}
		if(Globals.DEBUG){
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
		if(Globals.DEBUG){
			System.out.println("[d] Started creation of input web");
		}
		if(inputWebCreated==true){
			throw new InputWebException();
		}
		inputWebCreated=true;
		int inputLayerNeuronCount = inputLayer.size();
		if(Globals.DEBUG){
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
		int i = 0;
		for(Neuron n : outputLayer){
			n.setInputCount(hiddenLayers.get(hiddenLayers.size()-1).size()); 
			n.setSingleWeight(i, 1);
			i++;
		}
		if(Globals.DEBUG){
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
		if(Globals.DEBUG){
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
		if(Globals.DEBUG){
			System.out.println("[d] Hidden layer added");
		}
	}
	/**
	 * Randomizes all the weights in this network's hidden layers (-1<x<1)
	 */
	public void randomizeAllHiddenLayerWeights(){
		if(Globals.DEBUG){
			System.out.println("[d] Randomizing hidden layer weights");
		}
		for(ArrayList<Neuron> hiddenLayer : hiddenLayers){
			for(Neuron neuron : hiddenLayer){
				neuron.setRandomWeights();
			}
		}
		if(Globals.DEBUG){
			System.out.println("[d] Randomization completed");
		}
	}
	/**
	 * Randomizes all the biases in this network's hidden layers (-100<x<100)
	 */
	public void randomizeAllHiddenLayerBiases(){
		if(Globals.DEBUG){
			System.out.println("[d] Randomizing hidden layer biases");
		}
		for(ArrayList<Neuron> hiddenLayer : hiddenLayers){
			for(Neuron neuron : hiddenLayer){
				neuron.setRandomBias();
			}
		}
		if(Globals.DEBUG){
			System.out.println("[d] Randomization completed");
		}
	}
	/**
	 * Takes an array of the same size as the input layer and stores it's values into the input layer
	 */
	public void initializeInputLayer(double[] input){
		if(Globals.DEBUG){
			System.out.println("[d] Overriding outputs for input layer neurons");
		}
		assert input.length==inputLayer.size();
		for(int i = 0; i < inputLayer.size(); i++){
			inputLayer.get(i).overrideOutput(input[i]);
		}
		if(Globals.DEBUG){
			System.out.println("[d] Override completed");
		}
	}
	public void computeOutput(){
		for(Neuron hiddenNeuron : hiddenLayers.get(0)){
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
		}
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
}
