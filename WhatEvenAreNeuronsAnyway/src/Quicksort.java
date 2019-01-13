//Code slightly modified from Cracking The Coding Interview's Github, found here:
//https://github.com/careercup/ctci/blob/master/java/Chapter%2011/Introduction/Quicksort.java
//Credit for the original Quicksort implementation goes to Gayle McDowell
public class Quicksort {
	public static void swap(NeuralNetwork[] array, int i, int j) {
		NeuralNetwork tmp = array[i];
		array[i] = array[j];
		array[j] = tmp;
	}
	
	public static int partition(NeuralNetwork arr[], int left, int right) {
		NeuralNetwork pivot = arr[(left + right) / 2]; // Pick a pivot point. Can be an element.
		
		while (left <= right) { // Until we've gone through the whole array
			// Find element on left that should be on right
			while (arr[left].getNetworkScore() < pivot.getNetworkScore()) { 
				left++;
			}
			
			// Find element on right that should be on left
			while (arr[right].getNetworkScore() > pivot.getNetworkScore()) {
				right--;
			}
			
			// Swap elements, and move left and right indices
			if (left <= right) {
				swap(arr, left, right);
				left++;
				right--;
			}
		}
		return left; 
	}
	
	public static void quickSort(NeuralNetwork arr[], int left, int right) {
		int index = partition(arr, left, right); 
		if (left < index - 1) { // Sort left half
			quickSort(arr, left, index - 1);
		}
		if (index < right) { // Sort right half
			quickSort(arr, index, right);
		}

	}
}