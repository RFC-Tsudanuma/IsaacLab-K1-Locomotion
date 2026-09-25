#include "vision_filter/ball_prediction.hpp"
#include "vision_filter/select_best_prediction.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

using namespace vision_filter;
using Result = BallPredictionResult;

struct Input {
    long long ns;
    double x, y, lx, ly;
    bool observed = true;
    bool transform_valid = true;
    BallPredictionInput value() const {
        BallPredictionInput v;
        v.stamp.sec = static_cast<int>(ns / 1000000000LL);
        v.stamp.nanosec = static_cast<unsigned>(ns % 1000000000LL);
        v.transform_valid = transform_valid;
        v.raw_ball.status = observed ? RawBallState::STATUS_OBSERVED : RawBallState::STATUS_LOST;
        v.raw_ball.global_position.x=x; v.raw_ball.global_position.y=y;
        v.raw_ball.local_position.x=lx; v.raw_ball.local_position.y=ly;
        return v;
    }
};

void number(double v) { if(std::isfinite(v)) std::cout<<v; else std::cout<<"null"; }
template<class T> void numbers(const T &values) {
    std::cout<<'['; bool first=true;
    for(auto v: values) { if(!first)std::cout<<','; first=false; number(v); }
    std::cout<<']';
}
void optional_number(const std::optional<double> &v) { if(v)number(*v);else std::cout<<"null"; }
void input_json(const Input &v) {
    std::cout<<"{\"stamp_ns\":"<<v.ns<<",\"global_xy\":["<<v.x<<','<<v.y
      <<"],\"local_xy\":["<<v.lx<<','<<v.ly<<"],\"observed\":"<<v.observed
      <<",\"transform_valid\":"<<v.transform_valid<<'}';
}
void result_json(const Result &r, bool public_snapshot=false) {
    std::cout<<"{\"status\":"<<static_cast<unsigned>(r.status)
       <<",\"measurement_accepted\":"<<r.measurement_accepted
       <<",\"velocity_valid\":"<<r.velocity.has_value()<<",\"state\":";
    if(r.position && r.velocity) numbers(std::array<double,4>{r.position->x,r.position->y,r.velocity->x,r.velocity->y});
    else if(public_snapshot) numbers(std::array<double,4>{}); else std::cout<<"null";
    std::cout<<",\"covariance\":";
    if(public_snapshot && !r.position) numbers(StateCovariance{}); else numbers(r.current_states_covariance);
    std::cout<<",\"innovation_log_likelihood_cm\":"; optional_number(r.innovation_log_likelihood);
    std::cout<<",\"confidence_internal\":"; number(r.confidence);
    std::cout<<",\"future_xy_0_1\":";
    if(r.future_position) numbers(std::array<double,2>{r.future_position->x,r.future_position->y}); else std::cout<<"null";
    std::cout<<",\"future_covariance_0_1\":";numbers(r.future_states_covariance);
    std::cout<<'}';
}

CvkfConfig config(int hypothesis) {
    const std::array<CvkfMotionModel,4> modes{CvkfMotionModel::kStationary,CvkfMotionModel::kRolling,CvkfMotionModel::kHighSpeed,CvkfMotionModel::kBounce};
    return CvkfConfig(.1,25.,80.,25.,250.,3.,modes.at(hypothesis),hypothesis==0?0.:1.,
      hypothesis==2?4.:0.,hypothesis==1?std::optional<double>{4.}:std::nullopt,1.,
      CvkfMeasurementCovariance{477.733905268419846,-4.131492504468594,153.060614974156124});
}
std::vector<double> transition() {
    std::vector<double> p(16,0.0166666666666667);
    for(int i=0;i<4;++i)p.at(i*4+i)=.95;
    return p;
}
struct Trace {
    Trace() {}
    bool stepped=false, accepted=false, bounce_seeded=false;
    int selected=-1;
    std::array<std::optional<double>,4> nis;
    std::array<Result,4> predictions;
    std::optional<Point> bounce_position;
    std::optional<Point> bounce_future;
    std::optional<StateCovariance> bounce_covariance;
};
struct Bank {
    std::vector<std::unique_ptr<CVKF>> filters;
    SelectBestPrediction selector{4,transition()};
    Trace trace;
    Bank() { for(int i=0;i<4;++i) filters.push_back(std::make_unique<CVKF>(config(i))); }
    void reset() { for(auto &f:filters)f->reset();selector.reset(); }
    bool gate(const Input &i) {
        if(!i.observed || !i.transform_valid) return false;
        bool accepted=false;
        for(int k=0;k<4;++k) {
            trace.nis.at(k)=filters.at(k)->normalized_innovation_squared(i.value());
            if(!trace.nis.at(k))return false;
            accepted=accepted || *trace.nis.at(k)<=9.21;
        }
        return accepted;
    }
    Result step(const Input &i,bool accept) {
        trace.stepped=true;trace.accepted=accept;
        auto effective=i.value();if(!accept)effective.raw_ball.status=RawBallState::STATUS_LOST;
        std::vector<Result> predictions;
        for(auto &f:filters)predictions.push_back(f->predict(effective));
        trace.selected=static_cast<int>(selector.select(predictions));
        for(int k=0;k<4;++k)trace.predictions.at(k)=predictions.at(k);
        const Result output=predictions.at(trace.selected);
        if(trace.selected!=3 && output.status!=FilteredBallState::STATUS_LOST)
            trace.bounce_seeded=filters.at(3)->seed_from_prediction(output,-1.);
        trace.bounce_position=filters.at(3)->future_position(0.);
        trace.bounce_future=filters.at(3)->future_position(.1);
        trace.bounce_covariance=filters.at(3)->future_position_covariance(0.);
        return output;
    }
};
void trace_json(const Trace &t) {
    std::cout<<"{\"stepped\":"<<t.stepped<<",\"accepted\":"<<t.accepted<<",\"selected\":"<<t.selected<<",\"nis\":[";
    for(int k=0;k<4;++k){if(k)std::cout<<',';optional_number(t.nis.at(k));}std::cout<<']';
    if(t.stepped){std::cout<<",\"hypotheses\":[";for(int k=0;k<4;++k){if(k)std::cout<<',';result_json(t.predictions.at(k));}std::cout<<']';}
    std::cout<<",\"bounce_seeded\":"<<t.bounce_seeded<<",\"bounce_post_state\":";
    if(t.bounce_position && t.bounce_future) numbers(std::array<double,4>{t.bounce_position->x,t.bounce_position->y,(t.bounce_future->x-t.bounce_position->x)/.1,(t.bounce_future->y-t.bounce_position->y)/.1}); else std::cout<<"null";
    std::cout<<",\"bounce_post_covariance\":";if(t.bounce_covariance)numbers(*t.bounce_covariance);else std::cout<<"null";
    std::cout<<'}';
}
struct NodeDriver {
    Bank confirmed,tentative;
    bool active=false;
    unsigned count=0;
    Trace confirmed_trace,tentative_trace;
    std::string event;
    void reset_tentative(){tentative.reset();count=0;}
    Result process(const Input &i) {
        confirmed.trace=Trace{};tentative.trace=Trace{};event="none";
        const bool observed=i.observed && i.transform_valid;
        std::optional<Result> existing;
        if(active){
            const bool accepted=observed && confirmed.gate(i);
            existing=confirmed.step(i,accepted);
            if(accepted && existing->measurement_accepted){reset_tentative();event="confirmed_accept";return *existing;}
            if(existing->status==FilteredBallState::STATUS_LOST){confirmed.reset();active=false;event="confirmed_lost";}
        }
        if(!observed){reset_tentative();event+="/missing";return existing.value_or(Result());}
        Result candidate;
        if(count==0){candidate=tentative.step(i,true);count=candidate.measurement_accepted?1:0;event+="/tentative_first";}
        else if(tentative.gate(i)){
            candidate=tentative.step(i,true);
            if(candidate.measurement_accepted)++count;else reset_tentative();
            event+="/tentative_accept";
        }else{
            reset_tentative();candidate=tentative.step(i,true);count=candidate.measurement_accepted?1:0;
            event+="/tentative_restart";
        }
        if(count>=2){
            // Preserve traces before swapping bank ownership for diagnostic output.
            confirmed_trace=confirmed.trace;tentative_trace=tentative.trace;
            std::swap(confirmed,tentative);active=true;count=0;tentative.reset();event+="/promote";
            return candidate;
        }
        return existing.value_or(Result());
    }
    void record(const Input&i,int env) {
        const auto output=process(i);
        if(event.find("/promote")==std::string::npos){confirmed_trace=confirmed.trace;tentative_trace=tentative.trace;}
        std::cout<<"{\"env\":"<<env<<",\"input\":";input_json(i);
        std::cout<<",\"event\":\""<<event<<"\",\"confirmed_active\":"<<active<<",\"capture_count\":"<<count<<",\"output\":";result_json(output,true);
        std::cout<<",\"confirmed_step\":";trace_json(confirmed_trace);std::cout<<",\"tentative_step\":";trace_json(tentative_trace);std::cout<<'}';
    }
};

Input sample(long long ns,double x,double y,bool observed=true,bool transform=true) {
    // Deliberately keep a nonzero, independently defined sensor range for R.
    return {ns,x,y,x-.25,y+.15,observed,transform};
}
std::vector<Input> movement() {
    std::vector<Input> data;
    for(int k=0;k<32;++k){
        double x=k<=15?1.+.16*k:1.+.16*15-.16*(k-15);
        double y=k<=15?-.4+.04*k:-.4+.04*15-.04*(k-15);
        const double jitter=(k%4==0?.006:k%4==1?-.004:k%4==2?.002:-.003);
        data.push_back(sample(1000000000LL+40000000LL*k,x+jitter,y-jitter/2));
    }
    return data;
}
std::vector<Input> acquisition() {
    return {sample(1000000000,1.,.2),sample(1040000000,1.03,.19),sample(1080000000,1.08,.18),
      sample(1120000000,1.12,.17,false),sample(1160000000,8.,-5.),sample(1200000000,8.02,-4.98),
      sample(1240000000,8.04,-4.96),sample(1280000000,1.,.2),sample(1320000000,8.08,-4.92),
      sample(4320000000,0.,0.,false),sample(4320000001,0.,0.,false),
      sample(4360000000,2.,1.),sample(4400000000,2.02,.99,false),sample(4440000000,2.04,.98),
      sample(4480000000,2.06,.97),sample(4520000000,2.08,.96,true,false),sample(4560000000,2.10,.95)};
}
int main(){
    std::cout<<std::setprecision(17)<<std::boolalpha;
    std::cout<<"{\"core_analytic\":[";
    bool analytic_first=true;
    for(int mode: {0,1}) {
        CVKF f(config(mode));
        for(const auto&i:std::vector<Input>{sample(1000000000,1.,0.),sample(1100000000,1.,0.,false)}) {
            if(!analytic_first)std::cout<<',';analytic_first=false;
            const auto r=f.predict(i.value());
            std::cout<<"{\"hypothesis\":"<<mode<<",\"input\":";input_json(i);std::cout<<",\"output\":";result_json(r);std::cout<<'}';
        }
    }
    std::cout<<"],\"core\":[";
    bool first=true;
    for(int mode: {0,1}){
        CVKF f(config(mode));auto data=acquisition();
        for(const auto&i:data){if(!first)std::cout<<',';first=false;const auto nis=f.normalized_innovation_squared(i.value());const auto r=f.predict(i.value());
            std::cout<<"{\"hypothesis\":"<<mode<<",\"input\":";input_json(i);std::cout<<",\"nis_before\":";optional_number(nis);std::cout<<",\"output\":";result_json(r);std::cout<<'}';}
    }
    std::cout<<"],\"bank\":[";Bank bank;first=true;
    for(const auto&i:movement()){if(!first)std::cout<<',';first=false;bank.trace=Trace{};bank.gate(i);const auto r=bank.step(i,true);
        std::cout<<"{\"input\":";input_json(i);std::cout<<",\"output\":";result_json(r);std::cout<<",\"trace\":";trace_json(bank.trace);std::cout<<'}';}
    std::cout<<"],\"node\":[";NodeDriver node;first=true;
    for(const auto&i:acquisition()){if(!first)std::cout<<',';first=false;node.record(i,0);}
    std::cout<<"],\"interleaved\":[";std::array<NodeDriver,3> nodes;first=true;auto moving=movement();auto acquire=acquisition();
    for(std::size_t k=0;k<acquire.size();++k)for(int env=0;env<3;++env){if(!first)std::cout<<',';first=false;
        Input i=env==0?acquire.at(k):moving.at(k);
        if(env==2){i.x+=.8;i.y-=.5;i.lx+=.8;i.ly-=.5;i.observed=(k%3!=1);}
        nodes.at(env).record(i,env);
    }
    std::cout<<"]}\n";
}
