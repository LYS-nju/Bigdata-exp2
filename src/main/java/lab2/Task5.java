package lab2;

import java.io.IOException;
import java.util.List;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

public class Task5 extends Configured implements Tool {
    static final class Stats {
        long n;
        final double[] sums = new double[Data.NUMERIC];
        final double[] squares = new double[Data.NUMERIC];
        final double[] bits = new double[Data.BINARY];

        void add(Data.Row r, double[] q) {
            n++;
            double[] x = Data.numeric(r);
            double[] b = Data.binary(r, q);
            for (int i = 0; i < x.length; i++) {
                sums[i] += x[i];
                squares[i] += x[i] * x[i];
            }
            for (int i = 0; i < b.length; i++) {
                bits[i] += b[i];
            }
        }

        void merge(String s) throws IOException {
            String[] c = s.split(",", -1);
            if (c.length != 1 + 2 * Data.NUMERIC + Data.BINARY) {
                throw new IOException("Invalid statistics shape");
            }
            n += Long.parseLong(c[0]);
            int k = 1;
            for (int i = 0; i < sums.length; i++) {
                sums[i] += Double.parseDouble(c[k]);
                k++;
            }
            for (int i = 0; i < squares.length; i++) {
                squares[i] += Double.parseDouble(c[k]);
                k++;
            }
            for (int i = 0; i < bits.length; i++) {
                bits[i] += Double.parseDouble(c[k]);
                k++;
            }
        }

        String encode() {
            StringBuilder b = new StringBuilder(Long.toString(n));
            for (int i = 0; i < sums.length; i++) {
                b.append(',');
                b.append(sums[i]);
            }
            for (int i = 0; i < squares.length; i++) {
                b.append(',');
                b.append(squares[i]);
            }
            for (int i = 0; i < bits.length; i++) {
                b.append(',');
                b.append(bits[i]);
            }
            return b.toString();
        }
    }

    public static class TrainMapper extends Mapper<LongWritable, Text, IntWritable, Text> {
        private double[] q;
        private final Stats[] stats = {new Stats(), new Stats()};

        @Override
        protected void setup(Context ctx) throws IOException {
            q = Data.thresholds(ctx.getConfiguration());
        }

        @Override
        protected void map(LongWritable key, Text value, Context ctx) {
            String line = value.toString();
            if (Data.skip(line)) {
                return;
            }
            Data.Row r = new Data.Row(line, true);
            stats[r.label].add(r, q);
            ctx.getCounter("LAB2", "MODEL_TRAIN_ROWS").increment(1);
        }

        @Override
        protected void cleanup(Context ctx) throws IOException, InterruptedException {
            for (int i = 0; i < 2; i++) {
                ctx.write(new IntWritable(i), new Text(stats[i].encode()));
            }
        }
    }

    public static class TrainCombiner extends Reducer<IntWritable, Text, IntWritable, Text> {
        @Override
        protected void reduce(IntWritable key, Iterable<Text> values, Context ctx) throws IOException, InterruptedException {
            Stats s = new Stats();
            for (Text v : values) {
                s.merge(v.toString());
            }
            ctx.write(key, new Text(s.encode()));
        }
    }

    public static class TrainReducer extends Reducer<IntWritable, Text, IntWritable, Text> {
        @Override
        protected void reduce(IntWritable key, Iterable<Text> values, Context ctx) throws IOException, InterruptedException {
            Stats s = new Stats();
            for (Text v : values) {
                s.merge(v.toString());
            }
            ctx.write(key, new Text("NB1," + s.encode()));
        }
    }

    static final class Model {
        final Stats[] classes = {new Stats(), new Stats()};
        final boolean[] present = new boolean[2];

        void load(String line) throws IOException {
            String[] kv = line.split("\t", 2);
            if (kv.length != 2 || !kv[1].startsWith("NB1,")) {
                throw new IOException("Invalid model version");
            }
            int label = Integer.parseInt(kv[0]);
            if (label < 0 || label > 1 || present[label]) {
                throw new IOException("Invalid/duplicate model class");
            }
            classes[label].merge(kv[1].substring(4));
            present[label] = true;
        }

        void validate() throws IOException {
            if (!present[0] || !present[1] || classes[0].n + classes[1].n <= 0) {
                throw new IOException("Empty/incomplete model");
            }
            for (int c = 0; c < 2; c++) {
                Stats s = classes[c];
                if (s.n < 0) {
                    throw new IOException("Negative class count");
                }
                for (int i = 0; i < s.sums.length; i++) {
                    double v = s.sums[i];
                    if (!Double.isFinite(v) || v < 0) {
                        throw new IOException("Invalid model sums");
                    }
                }
                for (int i = 0; i < s.squares.length; i++) {
                    double v = s.squares[i];
                    if (!Double.isFinite(v) || v < 0) {
                        throw new IOException("Invalid model squares");
                    }
                }
                for (int i = 0; i < s.bits.length; i++) {
                    double v = s.bits[i];
                    if (!Double.isFinite(v) || v < 0 || v > s.n) {
                        throw new IOException("Invalid binary counts");
                    }
                }
            }
        }

        int predict(Data.Row r, double[] q) {
            double[] x = Data.numeric(r);
            double[] b = Data.binary(r, q);
            double[] score = new double[2];
            long total = classes[0].n + classes[1].n;
            for (int c = 0; c < 2; c++) {
                Stats s = classes[c];
                if (s.n == 0) {
                    score[c] = Double.NEGATIVE_INFINITY;
                    continue;
                }
                score[c] = Math.log((double) s.n / total);
                for (int i = 0; i < x.length; i++) {
                    double mean = s.sums[i] / s.n;
                    double variance = s.squares[i] / s.n - mean * mean;
                    if (variance < 1e-6) {
                        variance = 1e-6;
                    }
                    double delta = x[i] - mean;
                    score[c] -= 0.5 * (Math.log(2 * Math.PI * variance) + delta * delta / variance);
                }
                for (int i = 0; i < b.length; i++) {
                    double p = (s.bits[i] + 1) / (s.n + 2.0);
                    if (b[i] == 1) {
                        score[c] += Math.log(p);
                    } else {
                        score[c] += Math.log1p(-p);
                    }
                }
            }
            if (score[1] > score[0]) {
                return 1;
            }
            return 0;
        }
    }

    public static class PredictMapper extends Mapper<LongWritable, Text, Text, IntWritable> {
        private double[] q;
        private final Model model = new Model();

        @Override
        protected void setup(Context ctx) throws IOException {
            q = Data.thresholds(ctx.getConfiguration());
            List<String> lines = Data.readCache(ctx.getConfiguration(), "model");
            for (int i = 0; i < lines.size(); i++) {
                model.load(lines.get(i));
            }
            model.validate();
        }

        @Override
        protected void map(LongWritable key, Text value, Context ctx) throws IOException, InterruptedException {
            String line = value.toString();
            if (Data.skip(line)) {
                return;
            }
            Data.Row r = new Data.Row(line, false);
            int predicted = model.predict(r, q);
            ctx.write(new Text(r.id), new IntWritable(predicted));
            ctx.getCounter("LAB2", "TEST_ROWS").increment(1);
            ctx.getCounter("LAB2", "PREDICTED_" + predicted).increment(1);
        }
    }

    @Override
    public int run(String[] args) throws Exception {
        if (args.length != 5) {
            System.err.println("Usage: hadoop jar target/credit-mapreduce-lab2.jar lab2.Task5 <train> <test> <task3Output> <modelOutput> <predictionOutput>");
            return 2;
        }

        Path predictions = new Path(args[4]);
        if (predictions.getFileSystem(getConf()).exists(predictions)) {
            throw new IllegalArgumentException("Prediction output exists: " + predictions);
        }
        Path test = new Path(args[1]);
        if (!test.getFileSystem(getConf()).exists(test)) {
            throw new IllegalArgumentException("Missing test input: " + test);
        }

        Job trainJob = Data.createJob(getConf(), "Lab2 Task5 - Naive Bayes training", args[0], args[3]);
        trainJob.setMapperClass(TrainMapper.class);
        trainJob.setCombinerClass(TrainCombiner.class);
        trainJob.setReducerClass(TrainReducer.class);
        trainJob.setMapOutputKeyClass(IntWritable.class);
        trainJob.setMapOutputValueClass(Text.class);
        trainJob.setOutputKeyClass(IntWritable.class);
        trainJob.setOutputValueClass(Text.class);
        trainJob.setNumReduceTasks(1);
        Data.cache(trainJob, args[2], "quartiles");

        if (!trainJob.waitForCompletion(true)) {
            return 1;
        }

        Job predictJob = Data.createJob(getConf(), "Lab2 Task5 - distributed prediction", args[1], args[4]);
        predictJob.setMapperClass(PredictMapper.class);
        predictJob.setMapOutputKeyClass(Text.class);
        predictJob.setMapOutputValueClass(IntWritable.class);
        predictJob.setOutputKeyClass(Text.class);
        predictJob.setOutputValueClass(IntWritable.class);
        predictJob.setNumReduceTasks(0);
        Data.cache(predictJob, args[2], "quartiles");
        Data.cache(predictJob, args[3], "model");

        if (!predictJob.waitForCompletion(true)) {
            return 1;
        }
        return 0;
    }

    public static void main(String[] args) throws Exception {
        System.exit(ToolRunner.run(new Configuration(), new Task5(), args));
    }
}
