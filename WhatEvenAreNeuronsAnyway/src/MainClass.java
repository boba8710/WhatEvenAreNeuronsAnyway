import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class MainClass {
	public static void main(String[] args){
		int imageWidth = 1280;
		int imageHeight = 720;
		int entryNeurons = imageWidth*imageHeight;
		int exitNeurons  = imageWidth*imageHeight;
		int totalHiddenLayers = 3;
		int neuronsPerHiddenLayer = 50;
		int annotatedDatasetSize = 75;
		double[][] scoringInput = new double[annotatedDatasetSize][entryNeurons];
		double[][] scoringOutput = new double[annotatedDatasetSize][exitNeurons];
		for(int i = 0; i < annotatedDatasetSize; i++){
			System.out.println("Reading from image: "+i);
			BufferedImage inputImg = null;
			BufferedImage resultImg = null;
			try {
				inputImg = ImageIO.read(new File("C:\\Users\\Zaine\\Desktop\\COMP SCI\\DataSets\\8 bit blackwhite classified\\image_"+i+".jpg"));
				resultImg = ImageIO.read(new File("C:\\Users\\Zaine\\Desktop\\COMP SCI\\DataSets\\8 bit blackwhite classified\\image_"+i+"_result.jpg"));
			} catch (IOException e) {
				e.printStackTrace();
			}
			int neuronIterator = 0;
			for(int x = 0; x < imageWidth;x++){
				for(int y = 0; y < imageHeight;y++){
					scoringInput[i][neuronIterator]=((inputImg.getRGB(x,y)&0xFF)/255)*2-1;
					scoringOutput[i][neuronIterator]=((resultImg.getRGB(x,y)&0xFF)/86)*2-1;
					neuronIterator++;
				}
			}
		}
		GradientDescent gradientDescent = new GradientDescent(entryNeurons, exitNeurons, totalHiddenLayers, neuronsPerHiddenLayer, 0.0001);
		gradientDescent.trainNetwork(scoringInput, scoringOutput, 1);
	}
}