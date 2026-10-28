package lab2;

import java.io.IOException;
import java.math.BigDecimal;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

public class Task2 extends Configured implements Tool {
    public static class AverageMapper extends Mapper<LongWritable, Text, Text, Text> {
        private final Map<String, String[]> orgs = new HashMap<String, String[]>();

        @Override
        protected void setup(Context ctx) throws IOException {
            List<String> lines = Data.readCache(ctx.getConfiguration(), "orgstats");
            for (int i = 0; i < lines.size(); i++) {
                String[] kv = lines.get(i).split("\t", 2);
                if (kv.length != 2) {
                    throw new IOException("Invalid/duplicate organization statistics");
                }
                String[] fields = kv[1].split(",", -1);
                String[] previous = orgs.put(kv[0], fields);
                if (previous != null) {
                    throw new IOException("Invalid/duplicate organization statistics");
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
            String[] stats = orgs.get(r.org);
            if (stats == null) {
                throw new IOException("Organization absent from task1: " + r.org);
            }
            BigDecimal count = new BigDecimal(stats[0]);
            BigDecimal sum = new BigDecimal(stats[1]);
            boolean upper = r.asset().multiply(count).compareTo(sum) >= 0;
            if (upper) {
                ctx.write(new Text(r.org), new Text("1,0"));
            } else {
                ctx.write(new Text(r.org), new Text("0,1"));
            }
        }
    }

    public static class AverageReducer extends Reducer<Text, Text, Text, Text> {
        @Override
        protected void reduce(Text key, Iterable<Text> values, Context ctx) throws IOException, InterruptedException {
            long upper = 0;
            long lower = 0;
            for (Text v : values) {
                String[] c = v.toString().split(",");
                upper += Long.parseLong(c[0]);
                lower += Long.parseLong(c[1]);
            }
            ctx.write(key, new Text(upper + "," + lower));
        }
    }

    @Override
    public int run(String[] args) throws Exception {
        if (args.length != 3) {
            System.err.println("Usage: hadoop jar target/credit-mapreduce-lab2.jar lab2.Task2 <train> <task1Output> <output>");
            return 2;
        }

        Job job = Data.createJob(getConf(), "Lab2 Task2 - organization mean comparison", args[0], args[2]);
        job.setMapperClass(AverageMapper.class);
        job.setCombinerClass(AverageReducer.class);
        job.setReducerClass(AverageReducer.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        job.setNumReduceTasks(1);
        Data.cache(job, args[1], "orgstats");

        if (!job.waitForCompletion(true)) {
            return 1;
        }
        return 0;
    }

    public static void main(String[] args) throws Exception {
        System.exit(ToolRunner.run(new Configuration(), new Task2(), args));
    }
}
