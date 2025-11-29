using Base;

namespace WpfApp.Presenter;

using NeuralNetwork.V1;

public class V1PresenterService : IPresenterService
{
    public INeuralNetwork GetDoodleGuesserNetwork()
    {
        var network = NeuralNetwork.Restore("test.nn");
        network.SetActivationType(ActivationType.RELU, ActivationType.SOFTMAX);
        network.SetCostType(CostType.BINARY_CROSS_ENTROPY);
        
        return network;
    }

    public (INeuralNetwork, ILearningState) GetGraphGuesserNetwork()
    {
        var network = new NeuralNetwork(2, 3, 2);
        return (network, new LearningState(network));
    }
}