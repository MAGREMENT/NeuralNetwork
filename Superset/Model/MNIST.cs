namespace Model;

public static class MNIST
{
    public static IReadOnlyList<(int, double[])> Read(string labelsFiles, string imagesFile, int limit = int.MaxValue)
    {
        using var labelsStream = new FileStream(labelsFiles, new FileStreamOptions
        {
            Access = FileAccess.Read,
            Mode = FileMode.Open,
            Share = FileShare.Read
        });
        using var imagesStream = new FileStream(imagesFile, new FileStreamOptions
        {
            Access = FileAccess.Read,
            Mode = FileMode.Open,
            Share = FileShare.Read
        });

        Span<byte> labelsHeader = stackalloc byte[8];
        var n = labelsStream.Read(labelsHeader);
        if (n != 8) throw new Exception("Error while reading labels header");

        Span<byte> imagesHeader = stackalloc byte[16];
        n = imagesStream.Read(imagesHeader);
        if (n != 16) throw new Exception("Error while reading images header");

        var count = GetFlippedInt(labelsHeader, 4);
        if (count != GetFlippedInt(imagesHeader, 4)) throw new Exception("Different images and labels count");

        count = Math.Min(limit, count);
        
        var rows = GetFlippedInt(imagesHeader, 8);
        var cols = GetFlippedInt(imagesHeader, 12);
        var total = rows * cols;
        
        var result = new (int, double[])[count];
        for(int i = 0; i < count; i++)
        {
            var labelBuffer = new byte[1];
            var imageBuffer = new byte[total];

            n = labelsStream.Read(labelBuffer, 0, 1);
            if (n != 1) throw new Exception($"Error while reading label #{i + 1}");

            n = imagesStream.Read(imageBuffer, 0, total);
            if(n != total) throw new Exception($"Error while reading image #{i + 1}");
            
            var label = (int)labelBuffer[0];
            var arr = new double[total];
            for (int j = 0; j < total; j++)
            {
                arr[j] = imageBuffer[j] / 255.0;
            }

            result[i] = (label, arr);
        }
        
        return result;
    }

    public static FlattenedData FlattenForNeuralNetwork(IReadOnlyList<(int, double[])> data)
    {
        var imgSize = data[0].Item2.Length;
        var inputs = new double[data.Count * imgSize];
        var expected = new double[10 * data.Count];

        for (int i = 0; i < data.Count; i++)
        {
            var value = data[i].Item1;
            expected[i * 10 + value] = 1;

            var img = data[i].Item2;
            var start = imgSize * i;
            for (int j = 0; j < imgSize; j++)
            {
                inputs[start + j] = img[j];
            }
        }

        return new FlattenedData(inputs, imgSize, expected, 10);
    }
    
    private static int GetFlippedInt(Span<byte> span, int from)
    {
        return span[from + 3] | (span[from + 2] << 8) | (span[from + 1] << 16) | (span[from] << 24);
    }
}