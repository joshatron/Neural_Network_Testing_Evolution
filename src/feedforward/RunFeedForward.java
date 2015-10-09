package feedforward;

/**
 * Created by joshua on 10/4/15.
 */
public class RunFeedForward
{
    /**
     * Defines a neural network and trains it on an array of examples.
     * @param examples array containing examples, last 2 are expected outputs
     * @param hiddenLayers number of hidden layers
     * @param sizesLayers array containing sizes of each layer
     * @param learningRate learning rate for the net
     * @param momentum momentum for the net
     * @param activationFunction activation function to use
     * @param timesThrough number of times to run through the examples
     * @return the final net that was created
     */
    public static FeedForwardNeuralNetwork run(double[][] examples, int hiddenLayers, int[] sizesLayers,
                                               double learningRate, double momentum,
                                               ActivationFunction activationFunction, int timesThrough)
    {
        FeedForwardNeuralNetwork net = new FeedForwardNeuralNetwork(hiddenLayers, sizesLayers, activationFunction,
                                                                    momentum, learningRate);
        //run through the data set however many times
        for(int t = 0; t < timesThrough; t++)
        {
            //for each examples
            for(int k = 0; k < examples.length; k++)
            {
                //extract input and output
                if(k % 1000 == 0)
                {
                    System.out.println(Math.round(((k / examples.length) * 100)) + "% done");
                }
                double[] input = new double[examples[0].length - 2];
                double[] output = new double[2];

                for(int a = 0; a < input.length; a++)
                {
                    input[a] = examples[k][a];
                }
                output[0] = examples[k][examples[0].length - 2];
                output[1] = examples[k][examples[0].length - 1];

                //run backprop
                net.backprop(input, output);

            }
        }

        return net;
    }
}
