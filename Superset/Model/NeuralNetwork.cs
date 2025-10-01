using System.Runtime.InteropServices;

namespace Model;

public partial class NeuralNetwork : IDisposable
{
    private NeuralNetworkParameters _params;

    internal IntPtr Ptr { get; }
    public int Length { get; }

    public NeuralNetwork(NeuralNetworkParameters parameters, params int[] layers)
    {
        Ptr = Create(layers.Length, layers);
        Length = GetCount(Ptr);
        
        _params = parameters;
        ApplyParams(Ptr, _params);
    }

    private NeuralNetwork(IntPtr ptr, NeuralNetworkParameters parameters)
    {
        Ptr = ptr;
        _params = parameters;
        ApplyParams(Ptr, parameters);
        Initialize(Ptr);
    }

    public static NeuralNetwork Import(string file)
    {
        var p = new NeuralNetworkParameters();
        var ptr = FromFile(file, ref p);
        return new NeuralNetwork(ptr, p);
    }

    public void Save(string file)
    {
        Save(Ptr, _params, file);
    }

    public void ChangeParameters(EditParams action)
    {
        action(ref _params);
        ApplyParams(Ptr, _params);
    }

    public int GetInCount(int layer)
    {
        if (layer < 0 || layer >= GetCount(Ptr)) throw new IndexOutOfRangeException();
        return GetInCount(Ptr, layer);
    }

    public int GetInputInCount()
    {
        return GetInCount(Ptr, 0);
    }

    public int GetOutCount(int layer)
    {
        if (layer < 0 || layer >= GetCount(Ptr)) throw new IndexOutOfRangeException();
        return GetOutCount(Ptr, layer);
    }

    public int GetOutputOutCount()
    {
        return GetOutCount(Ptr, Length - 1);
    }

    public void SetWeight(int layer, int input, int output, double value)
    {
        CheckBounds(layer, input, output);
        SetWeight(Ptr, layer, input, output, value);
    }

    public double GetWeight(int layer, int input, int output)
    {
        CheckBounds(layer, input, output);
        return GetWeight(Ptr, layer, input, output);
    }
    
    public void SetBias(int layer, int output, double value)
    {
        if (output < 0 || output >= GetOutCount(layer)) throw new IndexOutOfRangeException();
        SetBias(Ptr, layer, output, value);
    }
    
    public double GetBias(int layer, int output)
    {
        if (output < 0 || output >= GetOutCount(layer)) throw new IndexOutOfRangeException();
        return GetBias(Ptr, layer, output);
    }

    public double[] Predict(double[] inputs) => Predict(inputs, inputs.Length);
    
    public double[] Predict(double[] inputs, int length)
    {
        if (length != GetInCount(0))
            throw new ArgumentException("Inputs length does not correspond to the first layer of the neural network");
        
        var count = GetOutCount(Length - 1);
        var arr = new double[count]; //TODO Look into using stackalloc when output is a single int
        
        Predict(Ptr, inputs, length, arr, count);
        return arr;
    }

    public void Learn(FlattenedData data, int batchSize, int iterations, LearningState? state = null)
    {
        var statePtr = state?.Ptr ?? IntPtr.Zero;
        Learn(Ptr, statePtr, data.Inputs, data.InputCutOff, data.Expected, data.ExpectedCutOff, data.GetCount(),
            batchSize, iterations);
    }

    public double GetCost(double[] inputs, double[] expected)
    {
        CheckTestDataBounds(inputs.Length, expected.Length);
        return Cost(Ptr, inputs, inputs.Length, expected, expected.Length); 
    }

    public double GetCost(FlattenedData data)
    {
        CheckTestDataBounds(data.InputCutOff, data.ExpectedCutOff);
        return MultiCost(Ptr, data.Inputs, data.InputCutOff, data.Expected, data.ExpectedCutOff, data.GetCount());
    }
    
    public void Dispose()
    {
        GC.SuppressFinalize(this);
        Dispose(Ptr);
    }

    private void CheckBounds(int layer, int input, int output)
    {
        if (input < 0 || input >= GetInCount(layer) || output < 0 || output >= GetOutCount(layer))
            throw new IndexOutOfRangeException();
    }

    private void CheckTestDataBounds(int inputCount, int outputCount)
    {
        if (inputCount != GetInCount(0) || outputCount != GetOutCount(Length - 1))
            throw new ArgumentException("Test data input or output count incorrect");
    }

    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial IntPtr Create(int count, int[] layers);

    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void Dispose(IntPtr ptr);

    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void ApplyParams(IntPtr ptr, NeuralNetworkParameters parameters);

    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial int GetCount(IntPtr ptr);
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial int GetOutCount(IntPtr ptr, int layer);
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial int GetInCount(IntPtr ptr, int layer);
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void SetWeight(IntPtr ptr, int layer, int input, int output, double value);
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial double GetWeight(IntPtr ptr, int layer, int input, int output);
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void SetBias(IntPtr ptr, int layer, int output, double value);
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial double GetBias(IntPtr ptr, int layer, int output);
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void Predict(IntPtr ptr, 
        [In, Out, MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 2)] double[] input, int inCount,
        [In, Out, MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 4)] double[] output, int outCount);

    [LibraryImport("libExport.dll", StringMarshalling = StringMarshalling.Utf8)]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial IntPtr FromFile(string file, ref NeuralNetworkParameters parameters);
    
    [LibraryImport("libExport.dll", StringMarshalling = StringMarshalling.Utf8)]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void Save(IntPtr ptr, NeuralNetworkParameters parameters, string file);
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void Initialize(IntPtr ptr);
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void Learn(IntPtr ptr, IntPtr learningState,
        [MarshalAs(UnmanagedType.LPArray)] double[] inputs, int inputCutOff, 
        [MarshalAs(UnmanagedType.LPArray)] double[] expected, 
        int expectedCutOff, int count, int batchSize, int iterations);
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial double Cost(IntPtr ptr, double[] inputs, int inputCount, double[] expected,
        int expectedCount);

    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial double MultiCost(IntPtr ptr, double[] inputs, int inputCutOff, double[] expected,
        int expectedCutOff, int count);
}

public delegate void EditParams(ref NeuralNetworkParameters p);

[StructLayout(LayoutKind.Sequential)]
public struct NeuralNetworkParameters
{
    public double InitialLearningRate;
    public double LearningRateDecay;
    public double Regularization;
    public double Momentum;
    public int ActivationType;
    public int OutputActivationType;
    public int CostType;

    public static NeuralNetworkParameters NoMomentumSigmoid { get; } = new()
    {
        InitialLearningRate = 10,
        LearningRateDecay = 0,
        Regularization = 0,
        Momentum = 0,
        ActivationType = Model.ActivationType.SIGMOID,
        OutputActivationType = Model.ActivationType.SIGMOID,
        CostType = Model.CostType.MEAN_SQUARED
    };
    
    public static NeuralNetworkParameters MomentumSigmoid { get; } = new()
    {
        InitialLearningRate = 1,
        LearningRateDecay = 0,
        Regularization = 0.1,
        Momentum = 0.9,
        ActivationType = Model.ActivationType.SIGMOID,
        OutputActivationType = Model.ActivationType.SIGMOID,
        CostType = Model.CostType.MEAN_SQUARED
    };
    
    public static NeuralNetworkParameters MomentumRelUSoftmax { get; } = new()
    {
        InitialLearningRate = 0.005,
        LearningRateDecay = 0.075,
        Regularization = 0.1,
        Momentum = 0.9,
        ActivationType = Model.ActivationType.RELU,
        OutputActivationType = Model.ActivationType.SOFTMAX,
        CostType = Model.CostType.CROSS_ENTROPY
    };
}

public static class ActivationType
{
    public const int DEFAULT = 0;
    public const int SIGMOID = 1;
    public const int TANH = 2;
    public const int RELU = 3;
    public const int SILU = 4;
    public const int SOFTMAX = 5;
}

public static class CostType
{
    public const int MEAN_SQUARED = 0;
    public const int CROSS_ENTROPY = 1;
}