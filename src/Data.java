package lab2;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.math.BigDecimal;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

final class Data {
    private Data() {
    }

    static final String[] TIERS = {"低资产", "中等资产", "较高资产", "高资产"};
    static final String[] CATEGORIES = {"存款", "理财", "私人理财", "基金", "国债", "银保通", "贵金属"};
    static final int NUMERIC = 10;
    static final int BINARY = 6;

    static final class Row {
        final String id;
        final String org;
        final int label;
        final BigDecimal[] money = new BigDecimal[NUMERIC];
        final boolean loan;
        final boolean cred;

        Row(String line, boolean train) {
            String[] c = line.split(",", -1);
            int expected = 14;
            if (train) {
                expected = 15;
            }
            if (c.length != expected) {
                throw new IllegalArgumentException("Unexpected CSV column count: " + c.length);
            }
            id = c[0].trim();
            org = c[1].trim();
            if (id.isEmpty() || org.isEmpty()) {
                throw new IllegalArgumentException("Missing customer/organization ID");
            }
            if (train) {
                label = Integer.parseInt(c[2].trim());
            } else {
                label = -1;
            }
            if (train) {
                if (label != 0 && label != 1) {
                    throw new IllegalArgumentException("bad_good must be 0 or 1");
                }
            }
            int start = 2;
            if (train) {
                start = 3;
            }
            for (int i = 0; i < NUMERIC; i++) {
                money[i] = amount(c[start + i]);
            }
            loan = flag(c[start + 10]);
            cred = flag(c[start + 11]);
        }

        BigDecimal asset() {
            return money[7];
        }

        BigDecimal debt() {
            return money[8];
        }
    }

    static boolean skip(String line) {
        String s = line.trim();
        if (s.startsWith("\uFEFF")) {
            s = s.substring(1);
        }
        if (s.isEmpty() || s.startsWith("CUST_ID,")) {
            return true;
        }
        return false;
    }

    static BigDecimal amount(String s) {
        s = s.trim();
        boolean missing = s.isEmpty() || s.equalsIgnoreCase("null") || s.equalsIgnoreCase("nan") || s.equalsIgnoreCase("NA");
        if (missing) {
            return BigDecimal.ZERO;
        }
        BigDecimal n = new BigDecimal(s);
        if (!Double.isFinite(n.doubleValue())) {
            throw new IllegalArgumentException("Non-finite amount");
        }
        return n;
    }

    static boolean flag(String s) {
        s = s.trim();
        if (s.equalsIgnoreCase("Y") || s.equals("1")) {
            return true;
        }
        if (s.isEmpty() || s.equalsIgnoreCase("N") || s.equals("0") || s.equalsIgnoreCase("null")) {
            return false;
        }
        throw new IllegalArgumentException("Unknown flag: " + s);
    }

    static String plain(BigDecimal x) {
        return x.stripTrailingZeros().toPlainString();
    }

    static Job createJob(Configuration conf, String name, String input, String output) throws IOException {
        Configuration jobConf = new Configuration(conf);
        Path in = new Path(input);
        Path out = new Path(output);
        if (!in.getFileSystem(jobConf).exists(in)) {
            throw new IllegalArgumentException("Missing input: " + in);
        }
        if (out.getFileSystem(jobConf).exists(out)) {
            throw new IllegalArgumentException("Output already exists; use a new run directory: " + out);
        }
        Job job = Job.getInstance(jobConf, name);
        job.setJarByClass(Data.class);
        FileInputFormat.addInputPath(job, in);
        FileOutputFormat.setOutputPath(job, out);
        return job;
    }

    static List<Path> parts(Configuration conf, String directory) throws IOException {
        Path p = new Path(directory);
        FileSystem fs = p.getFileSystem(conf);
        FileStatus[] statuses = fs.globStatus(new Path(p, "part-*"));
        if (statuses == null || statuses.length == 0) {
            throw new IOException("No part files: " + p);
        }
        List<Path> result = new ArrayList<Path>();
        for (int i = 0; i < statuses.length; i++) {
            if (statuses[i].isFile()) {
                result.add(statuses[i].getPath());
            }
        }
        Collections.sort(result);
        if (result.isEmpty()) {
            throw new IOException("No part files: " + p);
        }
        return result;
    }

    static void cache(Job job, String directory, String prefix) throws IOException {
        List<Path> paths = parts(job.getConfiguration(), directory);
        job.getConfiguration().setInt(prefix + ".count", paths.size());
        for (int i = 0; i < paths.size(); i++) {
            Path p = paths.get(i);
            String alias = prefix + "-" + i;
            job.getConfiguration().set(prefix + "." + i, alias);
            job.addCacheFile(URI.create(p.toUri().toString() + "#" + alias));
        }
    }

    static List<String> readCache(Configuration conf, String prefix) throws IOException {
        int n = conf.getInt(prefix + ".count", 0);
        if (n == 0) {
            throw new IOException("Missing distributed cache: " + prefix);
        }
        List<String> lines = new ArrayList<String>();
        for (int i = 0; i < n; i++) {
            String name = conf.get(prefix + "." + i);
            BufferedReader reader = open(new FileInputStream(name));
            try {
                String s = reader.readLine();
                while (s != null) {
                    if (!s.trim().isEmpty()) {
                        lines.add(s);
                    }
                    s = reader.readLine();
                }
            } finally {
                reader.close();
            }
        }
        return lines;
    }

    static long totalCount(Configuration conf, String task1) throws IOException {
        long n = 0;
        List<Path> files = parts(conf, task1);
        for (int i = 0; i < files.size(); i++) {
            Path p = files.get(i);
            BufferedReader reader = open(p.getFileSystem(conf).open(p));
            try {
                String s = reader.readLine();
                while (s != null) {
                    if (!s.isEmpty()) {
                        n += Long.parseLong(s.split("\t", 2)[1].split(",", -1)[0]);
                    }
                    s = reader.readLine();
                }
            } finally {
                reader.close();
            }
        }
        if (n <= 0) {
            throw new IOException("Empty training set");
        }
        return n;
    }

    private static BufferedReader open(InputStream in) {
        return new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8));
    }

    static double[] thresholds(Configuration conf) throws IOException {
        List<String> lines = readCache(conf, "quartiles");
        if (lines.size() != 1) {
            throw new IOException("Expected one quartile row");
        }
        String[] c = lines.get(0).split("\t");
        if (c.length != 3) {
            throw new IOException("Invalid quartile row");
        }
        String[] names = {"Q25:", "Q50:", "Q75:"};
        double[] q = new double[3];
        for (int i = 0; i < 3; i++) {
            if (!c[i].startsWith(names[i])) {
                throw new IOException("Invalid quartile name");
            }
            q[i] = Double.parseDouble(c[i].substring(4));
        }
        if (!Double.isFinite(q[0]) || !Double.isFinite(q[1]) || !Double.isFinite(q[2]) || q[0] > q[1] || q[1] > q[2]) {
            throw new IOException("Invalid quartiles");
        }
        return q;
    }

    static int tier(BigDecimal asset, double[] q) {
        for (int i = 0; i < 3; i++) {
            if (asset.compareTo(BigDecimal.valueOf(q[i])) <= 0) {
                return i;
            }
        }
        return 3;
    }

    static double[] numeric(Row r) {
        double[] x = new double[NUMERIC];
        for (int i = 0; i < x.length; i++) {
            double value = r.money[i].doubleValue();
            if (value < 0.0) {
                value = 0.0;
            }
            x[i] = Math.log1p(value);
        }
        return x;
    }

    static double[] binary(Row r, double[] q) {
        double[] x = new double[BINARY];
        if (r.loan) {
            x[0] = 1;
        } else {
            x[0] = 0;
        }
        if (r.cred) {
            x[1] = 1;
        } else {
            x[1] = 0;
        }
        x[2 + tier(r.asset(), q)] = 1;
        return x;
    }
}
