package lab2;

import java.io.IOException;
import java.math.BigDecimal;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

public class Task1 extends Configured implements Tool {
    public static class OrgMapper extends Mapper<LongWritable, Text, Text, Text> {
        @Override
        protected void map(LongWritable key, Text value, Context ctx) throws IOException, InterruptedException {
            String line = value.toString();
            if (Data.skip(line)) {
                return;
            }
            Data.Row r = new Data.Row(line, true);
            String out = "1," + Data.plain(r.asset()) + "," + Data.plain(r.debt()) + "," + r.label;
            ctx.write(new Text(r.org), new Text(out));
            ctx.getCounter("LAB2", "TRAIN_ROWS").increment(1);
        }
    }

    public static class OrgReducer extends Reducer<Text, Text, Text, Text> {
        @Override
        protected void reduce(Text key, Iterable<Text> values, Context ctx) throws IOException, InterruptedException {
            long n = 0;
            long bad = 0;
            BigDecimal a = BigDecimal.ZERO;
            BigDecimal d = BigDecimal.ZERO;
            for (Text v : values) {
                String[] c = v.toString().split(",", -1);
                n += Long.parseLong(c[0]);
                a = a.add(new BigDecimal(c[1]));
                d = d.add(new BigDecimal(c[2]));
                bad += Long.parseLong(c[3]);
            }
            ctx.write(key, new Text(n + "," + Data.plain(a) + "," + Data.plain(d) + "," + bad));
        }
    }

    @Override
    public int run(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println("Usage: hadoop jar target/credit-mapreduce-lab2.jar lab2.Task1 <train> <output>");
            return 2;
        }

        Job job = Data.createJob(getConf(), "Lab2 Task1 - organization statistics", args[0], args[1]);
        job.setMapperClass(OrgMapper.class);
        job.setCombinerClass(OrgReducer.class);
        job.setReducerClass(OrgReducer.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        job.setNumReduceTasks(1);

        if (!job.waitForCompletion(true)) {
            return 1;
        }
        return 0;
    }

    public static void main(String[] args) throws Exception {
        System.exit(ToolRunner.run(new Configuration(), new Task1(), args));
    }
}
