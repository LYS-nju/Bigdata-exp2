package lab2;

import java.io.IOException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

public class Task3 extends Configured implements Tool {
    public static class QuartileMapper extends Mapper<LongWritable, Text, DoubleWritable, LongWritable> {
        @Override
        protected void map(LongWritable key, Text value, Context ctx) throws IOException, InterruptedException {
            String line = value.toString();
            if (Data.skip(line)) {
                return;
            }
            Data.Row r = new Data.Row(line, true);
            ctx.write(new DoubleWritable(r.asset().doubleValue()), new LongWritable(1));
        }
    }

    public static class FrequencyCombiner extends Reducer<DoubleWritable, LongWritable, DoubleWritable, LongWritable> {
        @Override
        protected void reduce(DoubleWritable key, Iterable<LongWritable> values, Context ctx) throws IOException, InterruptedException {
            long n = 0;
            for (LongWritable v : values) {
                n += v.get();
            }
            ctx.write(key, new LongWritable(n));
        }
    }

    public static class QuartileReducer extends Reducer<DoubleWritable, LongWritable, Text, NullWritable> {
        private long total;
        private long seen;
        private final long[] ranks = new long[3];
        private final double[] q = new double[3];

        @Override
        protected void setup(Context ctx) throws IOException {
            total = ctx.getConfiguration().getLong("lab2.total", -1);
            if (total <= 0) {
                throw new IOException("Invalid sample size");
            }
            for (int i = 0; i < 3; i++) {
                ranks[i] = (total * (i + 1) + 3) / 4;
            }
        }

        @Override
        protected void reduce(DoubleWritable key, Iterable<LongWritable> values, Context ctx) {
            long n = 0;
            for (LongWritable v : values) {
                n += v.get();
            }
            for (int i = 0; i < 3; i++) {
                if (seen < ranks[i] && seen + n >= ranks[i]) {
                    q[i] = key.get();
                }
            }
            seen += n;
        }

        @Override
        protected void cleanup(Context ctx) throws IOException, InterruptedException {
            if (seen != total) {
                throw new IOException("Task1 count differs from input: " + total + " vs " + seen);
            }
            ctx.write(new Text("Q25:" + q[0] + "\tQ50:" + q[1] + "\tQ75:" + q[2]), NullWritable.get());
        }
    }

    @Override
    public int run(String[] args) throws Exception {
        if (args.length != 3) {
            System.err.println("Usage: hadoop jar target/credit-mapreduce-lab2.jar lab2.Task3 <train> <task1Output> <output>");
            return 2;
        }

        Job job = Data.createJob(getConf(), "Lab2 Task3 - globally sorted quartiles", args[0], args[2]);
        job.setMapperClass(QuartileMapper.class);
        job.setCombinerClass(FrequencyCombiner.class);
        job.setReducerClass(QuartileReducer.class);
        job.setMapOutputKeyClass(DoubleWritable.class);
        job.setMapOutputValueClass(LongWritable.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(NullWritable.class);
        job.setNumReduceTasks(1);
        job.getConfiguration().setLong("lab2.total", Data.totalCount(getConf(), args[1]));

        if (!job.waitForCompletion(true)) {
            return 1;
        }
        return 0;
    }

    public static void main(String[] args) throws Exception {
        System.exit(ToolRunner.run(new Configuration(), new Task3(), args));
    }
}
