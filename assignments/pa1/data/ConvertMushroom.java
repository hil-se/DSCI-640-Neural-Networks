package data;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.Map;
import java.util.HashSet;
import java.util.HashMap;

import util.Log;

public class ConvertMushroom {
    public static void main(String[] arguments) {
        try {
            //create a buffered reader given the filename (which requires creating a File and FileReader object beforehand)
            BufferedReader bufferedReader = new BufferedReader(new FileReader(new File("./datasets/agaricus-lepiota.data")));
            BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(new File("./datasets/agaricus-lepiota.txt")));

            String readLine = "";
            //read the file line by line
            List<String[]> data = new ArrayList<>();
            while ((readLine = bufferedReader.readLine()) != null) {
                Log.info(readLine); //print out each line of the file if the -DLOG_LEVEL=DEBUG system property is set on the command line

                if (readLine.length() == 0 || readLine.charAt(0) == '#') {
                    //empty lines are skipped, as well as lines beginning
                    //with the '#' character, these are comments and also skipped
                    continue;
                }

                String[] values = readLine.split(",");
                data.add(values);
            }
            int n = data.get(0).length;
            String[] labels = new String[data.size()];
            for (int i=0; i<data.size();i++){
                if (data.get(i)[0].equals("p")) {
                    labels[i] = "1"; //this will be the first class
                } else if (data.get(i)[0].equals("e")) {
                    labels[i] = "0"; //this will be the second class
                } else {
                    System.err.println("ERROR: unknown class in agaricus-lepiota.data file: '" + data.get(i)[0] + "'");
                    System.err.println("This should not happen.");
                    System.exit(1);
                }
            }
            List<String[]> oneHotData = new ArrayList<>();
            for (int j=1; j<n; j++){
                Set<String> unique = new HashSet<>();
                for (int i=0; i<data.size();i++){
                    unique.add(data.get(i)[j]);
                }
                String[] uniqueArray = unique.toArray(new String[0]);
                Map<String, String> map = new HashMap<>();
                for (int k=0; k<uniqueArray.length; k++){
                    String[] oneHot = new String[uniqueArray.length];
                    for (int q=0; q<uniqueArray.length; q++){
                        if (q==k){
                            oneHot[q] = "1";
                        }
                        else {
                            oneHot[q] = "0";
                        }
                    }
                    map.put(uniqueArray[k], String.join(",", oneHot));
                }
                String[] newData = new String[data.size()];
                for (int i=0; i<data.size();i++){
                    newData[i] = map.get(data.get(i)[j]);
                }
                oneHotData.add(newData);
            }
            for (int i=0; i<data.size();i++){
                String[] newLine = new String[n-1];
                StringBuffer sb = new StringBuffer();
                sb.append(labels[i]);
                sb.append(":");
                for (int j=0; j<n-1; j++){
                    newLine[j] = oneHotData.get(j)[i];
                }
                sb.append(String.join(",", newLine));
                sb.append("\n");
                Log.info(sb.toString());
                bufferedWriter.write(sb.toString());
            }
            bufferedWriter.close();
            bufferedReader.close();
        } catch (IOException e) {
            Log.fatal("ERROR converting iris data file");
            e.printStackTrace();
            System.exit(1);
        }
    }
}
