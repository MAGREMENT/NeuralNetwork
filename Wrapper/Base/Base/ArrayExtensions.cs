namespace Base;

public static class ArrayExtensions
{
    public static T[] Copy<T>(this T[] arr)
    {
        var result = new T[arr.Length];
        arr.CopyTo(result, 0);
        return result;
    }
    
    public static T[] Copy<T>(this T[] arr, int from, int to)
    {
        var result = new T[to - from];
        Array.Copy(arr, from, result, 0, result.Length);
        return result;
    }

    public static T[,] To2D<T>(this T[] arr, int width, int height)
    {
        var result = new T[width, height];

        for (int i = 0; i < width; i++)
        {
            for (int j = 0; j < height; j++)
            {
                result[i, j] = arr[j * width + i];
            }
        }

        return result;
    }

    public static void Shuffle<T>(this T[] array, int count)
    {
        var random = new Random();
        for (int i = 0; i < count; i++)
        {
            var to = random.Next(array.Length);

            (array[i], array[to]) = (array[to], array[i]);
        }
    }

    public static int IndexOfHighestValue(this double[] arr)
    {
        if (arr.Length == 0) return -1;

        var max = arr[0];
        var ind = 0;

        for (int i = 1; i < arr.Length; i++)
        {
            if (arr[i] > max)
            {
                ind = i;
                max = arr[i];
            }
        }

        return ind;
    }
    
    public static int IndexOfHighestValue(this double[] arr, int from, int to)
    {
        var length = to - from;
        if (length == 0) return -1;

        var max = arr[from];
        var ind = 0;

        for (int i = 1; i < length; i++)
        {
            if (arr[from + i] > max)
            {
                ind = i;
                max = arr[from + i];
            }
        }

        return ind;
    }
}