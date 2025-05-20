using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using AI;
using System.Collections.Generic;
using System;
using System.Linq;

namespace ImgRecon2
{
    public class Main : Game
    {
        private GraphicsDeviceManager graphics;
        private SpriteBatch spriteBatch;

        private NN2 nn;
        List<Image> imgs;
        List<Image> imgsTrain;
        private const int DataSize = 784;
        private const int DimSize = 28;
        private const int epochs = 1;
        private const int miniBatchSize = 32;
        private const float learningRate = 0.001f;
        private const float movingAvgBeta = 0.9f;
        private static int[] networkForm = new int[] { DataSize, 64, 32, 10 };

        public Main()
        {
            graphics = new GraphicsDeviceManager(this);
            Content.RootDirectory = "Content";
            IsMouseVisible = true;
        }

        protected override void Initialize()
        {
            imgs = MnistReader.ReadTrainingData().ToList();
            nn = new NN2(networkForm, learningRate, movingAvgBeta);

            float[][] inputs = new float[miniBatchSize][];
            float[][] targets = new float[miniBatchSize][];
            for(int e = 0; e < epochs; e++)
            {
                for (int i = 0; i < imgs.Count; i++)
                {
                    inputs[i % miniBatchSize] = new float[DataSize];
                    targets[i % miniBatchSize] = new float[10];
                    targets[i % miniBatchSize][imgs[i].Label] = 1f;
                    for(int x = 0; x < DimSize; x++)
                        for(int y = 0; y < DimSize; y++)
                            inputs[i % miniBatchSize][x * DimSize + y] = imgs[i].Data[x, y] / 255f;

                    if (i % miniBatchSize == miniBatchSize - 1)
                    {
                        Console.WriteLine(i);
                        nn.Train(inputs, targets);
                    }
                }
            }

            imgsTrain = MnistReader.ReadTrainingData().ToList();
            int score = 0;
            float[] input = new float[DataSize];
            for (int i = 0; i < imgsTrain.Count; i++)
            {
                for (int x = 0; x < DimSize; x++)
                    for (int y = 0; y < DimSize; y++)
                        input[x * DimSize + y] = imgsTrain[i].Data[x, y] / 255f;

                float[] ff = nn.FeedForward(input);
                if (ArgMax(ff) == imgsTrain[i].Label)
                    score++;
            }

            Console.WriteLine($"Correct on training data {(float)score / imgsTrain.Count}% of the time");


            base.Initialize();
        }

        private static int ArgMax(float[] data)
        {
            float max = data[0];
            int maxIndex = 0;

            for(int i = 1; i < data.Length; i++)
            {
                if (data[i] > max)
                {
                    max = data[i];
                    maxIndex = i;
                }
            }

            return maxIndex;
        }

        protected override void LoadContent()
        {
            spriteBatch = new SpriteBatch(GraphicsDevice);

            // TODO: use this.Content to load your game content here
        }

        protected override void Update(GameTime gameTime)
        {
            if (GamePad.GetState(PlayerIndex.One).Buttons.Back == ButtonState.Pressed || Keyboard.GetState().IsKeyDown(Keys.Escape))
                Exit();

            // TODO: Add your update logic here

            base.Update(gameTime);
        }

        protected override void Draw(GameTime gameTime)
        {
            GraphicsDevice.Clear(Color.CornflowerBlue);

            // TODO: Add your drawing code here

            base.Draw(gameTime);
        }
    }
}