clear
load('D:\\Research\\Interdisciplinary Schools\\Research Groups\\Team9-2025\\data\\sessions.mat')
projPath = 'E:/Database/toShare/toShare/';
addpath(genpath([projPath '/codingLib']))
session_names = [63 64 67 68 69 71 74 76 79 80 84 95 97 98 99 109 110 111 112];

%%
num_sessions = length(session_names);
bin_size = 200;
pre_time = -3000;  
post_time = 2000;
pre_dur = round(pre_time / bin_size);
post_dur = round(post_time / bin_size);
num_position_vars = 4;

task_vars = cell(1, num_sessions);
position_vars = cell(1, num_sessions);
spikes = cell(1, num_sessions);
events = cell(1, num_sessions);

unit_names = cell(1, num_sessions);
channel_names = cell(1, num_sessions);
unit_types = cell(1, num_sessions);
task_var_names = {'tunp'; 'tslp'; 'rew_rate'; 'rew_ratio'; 'rew'; 'choice'; 'last_choice'};
position_var_names = {'x'; 'y'; 'd'; 'motion'};
event_names = {'b1_pushed_times'; 'b2_pushed_times'; 'b1_rew'; 'b2_rew'};

tunm = {};
for i = 1:num_sessions
    s = session_names(i);
    
    load([projPath 'data/binned/' sessionNames{s} '_binnedFr200_pre3post2.mat']);
    
    idx = iForag{s}( ...
        tslp{s}(iForag{s}) > 2 & tslp{s}(iForag{s}) < 60 );

    num_valid = length(idx);

    b1_rew = double(rew{s} == 1);
    b2_rew = double(rew{s} == 2);

    rew_ = min(rew{s}, 1);
    last_choice = [NaN choice{s}(1:end-1)];
    var_matrix = [
        tunp{s}(idx);
        tslp{s}(idx);
        rewRate{s}(idx);
        rewRatio{s}(idx);
        rew_(idx);
        choice{s}(idx);
        last_choice(idx);
    ];
    
    event_matrix = {
        b1PushedTimes{s};
        b2PushedTimes{s};
        b1_rew;
        b2_rew;
    };

    task_vars{i} = var_matrix;
    events{i} = event_matrix;
    
    tbLocX = trialCut(bLocX{s}, round(bPushedTimes{s} / bin_size), pre_dur, post_dur);
    tbLocY = trialCut(bLocY{s}, round(bPushedTimes{s} / bin_size), pre_dur, post_dur);
    tbLocD = trialCut(bLocD{s}, round(bPushedTimes{s} / bin_size), pre_dur, post_dur);
    tbMotion = trialCut(bMotion{s}, round(bPushedTimes{s} / bin_size), pre_dur, post_dur);
    
    position_vars{i} = zeros(num_position_vars, num_valid, size(tbLocX, 2));
    position_vars{i}(1, :, :) = tbLocX(idx, :);
    position_vars{i}(2, :, :) = tbLocY(idx, :);
    position_vars{i}(3, :, :) = tbLocD(idx, :);
    position_vars{i}(4, :, :) = tbMotion(idx, :);
    
    spikes{i} = btFr(:, idx, :);
    unit_names{i} = units;
    channel_names{i} = chan;
    unit_types{i} = unitTypes;

end

bin_times = trialTime(1:bin_size:end)/1000;

save([ pwd '/data.mat'], ...
    'num_sessions', 'session_names', 'spikes', 'unit_names', 'channel_names', 'unit_types', 'task_vars', 'task_var_names', 'position_vars', 'position_var_names', 'events', 'event_names', 'bin_times', 'bin_size');

disp(['Size of task_vars{1} (session ' num2str(session_names(1)) '): ' num2str(size(task_vars{1}))]);
disp(['Size of position_vars{1} (session ' num2str(session_names(1)) '): ' num2str(size(position_vars{1}))]);
disp(['Size of spikes{1} (session ' num2str(session_names(1)) '): ' num2str(size(spikes{1}))]);

%%
function tThing = trialCut(thing,events,pre_dur, post_dur)
    trialTime = pre_dur:post_dur;
    for i = 1:length(events)  
        ind = unique(round(events(i)+trialTime));
        ind2 = find(ind>0 & ind<length(thing));
        tThing(:,i,ind2) = thing(:,ind(ind2));
    end
    tThing = squeeze(tThing);
end