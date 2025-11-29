using System.Runtime.InteropServices;
using Base;

namespace NeuralNetwork.V2;

public partial class NeuralNetwork : INeuralNetwork
{
    private readonly IntPtr _ptr;

    public NeuralNetwork(IntPtr ptr)
    {
        _ptr = ptr;
    }

    public void Dispose()
    {
        Dispose(_ptr);
    }

    public IntPtr GetPointer() => _ptr;

    public int Length => GetLength(_ptr);
    public int InCount => GetInCount(_ptr);
    public int OutCount => GetOutCount(_ptr);

    public void RestoreParameters(string file)
    {
        LoadParameterFile(_ptr, file);
    }
    
    public double[] Predict(double[] input)
    {
        var result = new double[GetOutCount(_ptr)];
        Predict(_ptr, input, result);
        return result;
    }

    public double GetCost(FlattenedData data)
    {
        return Cost(_ptr, data.Inputs, data.Expected, data.GetCount());
    }

    public void Learn(FlattenedData data, int iterations, ILearningState? state)
    {
        if(state is null) LearnStateless(_ptr, data.Inputs, data.Expected, data.GetCount(), iterations);
        else Learn(_ptr, state.GetPointer(), data.Inputs, data.Expected, data.GetCount(), iterations);
    }
    
    [LibraryImport("Export.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void Dispose(IntPtr ptr);
    
    [LibraryImport("Export.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial int GetLength(IntPtr ptr);

    [LibraryImport("Export.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial int GetInCount(IntPtr ptr);
    
    [LibraryImport("Export.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial int GetOutCount(IntPtr ptr);

    [LibraryImport("Export.dll", StringMarshalling = StringMarshalling.Utf8)]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void LoadParameterFile(IntPtr ptr, string file);
    
    [LibraryImport("Export.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void Predict(IntPtr ptr, double[] inputs, double[] outputs);
    
    [LibraryImport("Export.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void Learn(IntPtr ptr, IntPtr state, double[] inputs, double[] expected, int count, int iterations);
    
    [LibraryImport("Export.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void LearnStateless(IntPtr ptr, double[] inputs, double[] expected, int count, int iterations);
    
    [LibraryImport("Export.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial double Cost(IntPtr ptr, double[] inputs, double[] expected, int count);
}