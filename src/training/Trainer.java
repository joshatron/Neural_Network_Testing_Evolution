package training;

import feedforward.FeedForwardNeuralNetwork;

/**
 * Created by joshua on 10/17/15.
 */
public interface Trainer
{
    FeedForwardNeuralNetwork run(FeedForwardNeuralNetwork net, double[][] examples);
}
