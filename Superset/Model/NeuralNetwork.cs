using System.Runtime.InteropServices;

namespace Model;

public class NeuralNetwork
{
    private readonly IntPtr _ptr;

    public NeuralNetwork()
    {
        _ptr = Initialize();
    }

    [DllImport("neural_network.dll")]
    private static extern IntPtr Initialize();
}