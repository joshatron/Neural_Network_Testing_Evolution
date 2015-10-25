package training;

import feedforward.FeedForwardNeuralNetwork;

public class MewLambdaEvolution implements Trainer
{
    double[] parameters;
    public MewLambdaEvolution(double[] parameters)
    {
        this.parameters = parameters;
    }

    @Override
    public FeedForwardNeuralNetwork run(FeedForwardNeuralNetwork net, double[][] examples)
    {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

}
