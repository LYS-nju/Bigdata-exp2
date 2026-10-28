package lab2;

import java.io.IOException;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.Arrays;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

public class Task4 extends Configured implements Tool {
    public static class TierMapper extends Mapper<LongWritable, Text, IntWritable, Text> {
        private double[] q;

        @Override
        protected void setup(Context ctx) throws IOException {
            q = Data.thresholds(ctx.getConfiguration());
            for (int i = 0; i < 4; i++) {
                try {
                    ctx.write(new IntWritable(i), new Text("0,0,0,0,0,0,0"));
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new IOException(e);
                }
            }
        }

        @Override
        protected void map(LongWritable key, Text value, Context ctx) throws IOException, InterruptedException {
            String line = value.toString();
            if (Data.skip(line)) {
                return;
            }
            Data.Row r = new Data.Row(line, true);
            StringBuilder b = new StringBuilder();
            for (int i = 0; i < 7; i++) {
                if (i > 0) {
                    b.append(',');
                }
                b.append(Data.plain(r.money[i]));
            }
            ctx.write(new IntWritable(Data.tier(r.asset(), q)), new Text(b.toString()));
        }
    }

    static BigDecimal[] sumSeven(Iterable<Text> values) {
        BigDecimal[] sums = new BigDecimal[7];
        Arrays.fill(sums, BigDecimal.ZERO);
        for (Text v : values) {
            String[] c = v.toString().split(",", -1);
            for (int i = 0; i < 7; i++) {
                sums[i] = sums[i].add(new BigDecimal(c[i]));
            }
        }
        return sums;
    }

    static int[] descendingOrder(BigDecimal[] ratios) {
        int count = ratios.length;
        boolean[] used = new boolean[count];
        int[] order = new int[count];
        for (int position = 0; position < count; position++) {
            int best = -1;
            for (int i = 0; i < count; i++) {
                if (used[i]) {
                    continue;
                }
                if (best < 0 || ratios[i].compareTo(ratios[best]) > 0) {
                    best = i;
                }
            }
            used[best] = true;
            order[position] = best;
        }
        return order;
    }

    public static class TierCombiner extends Reducer<IntWritable, Text, IntWritable, Text> {
        @Override
        protected void reduce(IntWritable key, Iterable<Text> values, Context ctx) throws IOException, InterruptedException {
            BigDecimal[] sums = sumSeven(values);
            StringBuilder b = new StringBuilder();
            for (int i = 0; i < 7; i++) {
                if (i > 0) {
                    b.append(',');
                }
                b.append(Data.plain(sums[i]));
            }
            ctx.write(key, new Text(b.toString()));
        }
    }

    public static class TierReducer extends Reducer<IntWritable, Text, Text, Text> {
        @Override
        protected void reduce(IntWritable key, Iterable<Text> values, Context ctx) throws IOException, InterruptedException {
            BigDecimal[] sums = sumSeven(values);
            BigDecimal total = BigDecimal.ZERO;
            for (int i = 0; i < 7; i++) {
                total = total.add(sums[i]);
            }
            BigDecimal[] ratios = new BigDecimal[7];
            for (int i = 0; i < 7; i++) {
                if (total.signum() == 0) {
                    ratios[i] = BigDecimal.ZERO;
                } else {
                    ratios[i] = sums[i].divide(total, 16, RoundingMode.HALF_UP);
                }
            }
            int[] order = descendingOrder(ratios);
            for (int i = 0; i < order.length; i++) {
                int category = order[i];
                String ratio = ratios[category].setScale(4, RoundingMode.HALF_UP).toPlainString();
                ctx.write(new Text(Data.TIERS[key.get()]), new Text(Data.CATEGORIES[category] + "\t" + ratio));
            }
        }
    }

    @Override
    public int run(String[] args) throws Exception {
        if (args.length != 3) {
            System.err.println("Usage: hadoop jar target/credit-mapreduce-lab2.jar lab2.Task4 <train> <task3Output> <output>");
            return 2;
        }

        Job job = Data.createJob(getConf(), "Lab2 Task4 - tier asset composition", args[0], args[2]);
        job.setMapperClass(TierMapper.class);
        job.setCombinerClass(TierCombiner.class);
        job.setReducerClass(TierReducer.class);
        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(Text.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        job.setNumReduceTasks(1);
        Data.cache(job, args[1], "quartiles");

        if (!job.waitForCompletion(true)) {
            return 1;
        }
        return 0;
    }

    public static void main(String[] args) throws Exception {
        System.exit(ToolRunner.run(new Configuration(), new Task4(), args));
    }
}
