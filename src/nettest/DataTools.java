package nettest;

/**
 *
 * @author David
 */
import java.io.IOException;
import java.io.File;
import java.io.PrintWriter;
import org.apache.commons.io.FileUtils;
import org.json.JSONArray;
import org.json.JSONObject;

public class DataTools {
    
    // Size of the input domain. Actual domain will be plus or minus 0.5 * domainSize.
    private static final float domainSize = 10.0f;

    /**
     * Create a JSON with a 2D array of doubles. All but the last double in each
     * array are inputs to the Rosenbrock function, and the last double is the
     * output. The range of the inputs is [-5,5).
     *
     * @param numInputs number of inputs for the Rosenbrock function
     * @param setSize   number of instances the output dataset will have
     * @return 2D array of the specified dataset.
     */
    public static double[][] generateData(int numInputs, int setSize) {
        JSONObject dataset = new JSONObject();
        double[][] data = new double[setSize][numInputs+1];
        for (int i = 0; i < setSize; i++) {
            double[] inputs = new double[numInputs]; // Next instance's inputs

            for (int j = 0; j < numInputs; j++) {   // Generate a random value for each input
                inputs[j] = Math.random() * domainSize - domainSize/2.0f;
                data[i][j] = inputs[j];
            }

            JSONArray instance = new JSONArray(inputs);

            // Set the final double to the output of the Rosenbrock function
            data[i][numInputs] = rosenbrock(inputs);
            instance = instance.put(data[i][numInputs]);

            String name = String.format("a%d", i);
            dataset.put(name, instance);
        }

        // Relative path. Must create a directory "~/data/"  to save the JSON.
        try (PrintWriter writer = new PrintWriter(String.format("data/%dD_Data_%d_vectors.JSON", numInputs, setSize), "UTF-8")) {
            writer.write(dataset.toString());
            writer.close();
        } catch (IOException e) {
            System.err.println(e.getMessage());
        }
        return data;
    }

    /**
     * Get a dataset from a JSON created by the generateData() method. If the
     * specified data set has not been created, generate one.
     * @param numInputs number of inputs used by the dataset
     * @param setSize   number of instances
     * @return          2D array data[i][j], where i = [0,setSize] and j = [0,numInputs]
     */
    public static double[][] getDataFromFile(int numInputs, int setSize) {
        // Relative path. Must create a directory "~/data/"  
        File file = new File(String.format("data/%dD_Data_%d_vectors.JSON", numInputs, setSize));
        double[][] data;

        try {
            String json = FileUtils.readFileToString(file);
            JSONObject jsonObject = new JSONObject(json);

            data = new double[setSize][numInputs + 1];

            for (int i = 0; i < setSize; i++) {
                JSONArray instance = jsonObject.getJSONArray(String.format("a%d", i));

                for (int j = 0; j < numInputs+1; j++) {
                    data[i][j] = instance.getDouble(j);
                }
            }
        } catch (IOException e) {
            System.out.println("Dataset does not exist. Creating a new dataset.");
            data = generateData(numInputs, setSize);
        }

        return data;
    }

    /**
     * Given an array of arbitrary size n, calculate the output of the Rosenbrock 
     * function that takes n variables.
     *
     * @param x array of inputs to the function
     * @return  result of the function
     */
    public static double rosenbrock(double[] x) {
        double result = 0;
        for (int i = 0; i < x.length - 1; i++) {
            result += Math.pow(1 - x[i], 2) + 100 * Math.pow((x[i + 1] - x[i] * x[i]), 2);
        }
        return result;
    }
    
    public static float getDomainSize() {
        return domainSize;
    }
}
