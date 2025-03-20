clc;clear;close all;

id = 1;
switch id
    case 1
        audioFile = "LPS-SI2homo-mH02-I04-USV";
    case 2
        audioFile = "LPS-SI2homo-mK01-L03-USV";
    case 3
        audioFile = "LPS-SI2homo-mL01-M02-USV";
    case 4
        audioFile = "LPS-SI2homo-mO02-S03-USV";
    case 5
        audioFile = "LPS-SI2homo-mR03-T01-USV";
    case 6
        audioFile = "LPS-SI2homo-mR04-S01-USV"; 
end
disp("------------------------");
disp(audioFile);


% File Paths
path = "/Users/gazda/Documents/CTU/Masters/Masters thesis/";
datapath = path + "data/";
audioPath = datapath + "usv_audio/denoise/" + audioFile + "_denoise.wav";
labelPath = datapath + "labels/" + audioFile + "-IVojt.txt";
outputPath = audioFile + "_detected.txt";
fprintf('Filename: %s\n', audioFile);


labels = PSDDetector(audioPath,'fs',5000);