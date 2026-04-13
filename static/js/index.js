window.HELP_IMPROVE_VIDEOJS = false;
/*
var INTERP_BASE = "./static/interpolation/stacked";
var NUM_INTERP_FRAMES = 240;

var interp_images = [];
function preloadInterpolationImages() {
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + '/' + String(i).padStart(6, '0') + '.jpg';
    interp_images[i] = new Image();
    interp_images[i].src = path;
  }
}

function setInterpolationImage(i) {
  var image = interp_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper').empty().append(image);
}

*/
$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    var options = {
			slidesToScroll: 1,
			slidesToShow: 1,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);
    preloadInterpolationImages();

    $('#interpolation-slider').on('input', function(event) {
      setInterpolationImage(this.value);
    });
    setInterpolationImage(0);
    $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);

    bulmaSlider.attach();

*/
})

function set_source(idx){
  data=[
    [
      "animation/image_35.png",
      "For this kitten, note that the paw, tail and head can move independently: in the first sample, the cat moves its nose forward to sniff the plant, whereas in the second, the paw moves in a waving motion.  <br/>Image credit <a href='https://unsplash.com/photos/93AcQpzcASE/'>Dim Hou</a>.",
      "animation/image_35-0-trajectories.mp4",
      "animation/image_35-0-warped.mp4",
      "animation/image_35-0-holefill.mp4",
      "animation/image_35-1-trajectories.mp4",
      "animation/image_35-1-warped.mp4",
      "animation/image_35-1-holefill.mp4",
    ],
    [
      "animation/image_64.png",
      "For this example of a person chopping vegetables, we see subtle motions to adjust the vegetables in the top sample, whereas the bottom example is a more aggressive cutting motion.  Although the hole filling model fails to deal with the larger occlusions in the second example, the overall motion of the hands is physically plausible.  In general, we find the model identifies hands surprisingly well.  <br/>Image credit <a href='https://unsplash.com/photos/uB7q7aipU2o/'>Jonathan Borba</a>.",
      "animation/image_64-0-trajectories.mp4",
      "animation/image_64-0-warped.mp4",
      "animation/image_64-0-holefill.mp4",
      "animation/image_64-1-trajectories.mp4",
      "animation/image_64-1-warped.mp4",
      "animation/image_64-1-holefill.mp4",
    ],
    [
      "animation/image_43.png",
      "This electrician is segmented well from the background, and shows two types of motion.  In the top example, the right hand pulls an object away from the wall, whereas in the bottom example, the left hand pushes an object into the wall as the camera moves.  <br/>Image credit <a href='https://unsplash.com/photos/-0-kl1BjvFc/'>Emmanuel Ikwuegbu</a>.",
      "animation/image_43-0-trajectories.mp4",
      "animation/image_43-0-warped.mp4",
      "animation/image_43-0-holefill.mp4",
      "animation/image_43-1-trajectories.mp4",
      "animation/image_43-1-warped.mp4",
      "animation/image_43-1-holefill.mp4",
    ],
    [
      "animation/image_26.png",
      "The TAP task is defined only for solid objects: point correspondences are not well-defined for liquids or gasses like clouds.  Despite this, it would seem that TAPIR can produce qualitatively reasonable tracks for them, as our diffusion model can predict plausible motion for them.  Here we see one example of clouds blowing to the left, and another blowing into the distance.  This suggests that the TAP task can generalize beyond the domains it was strictly intended for; i.e., it can serve as a scene representation even for objects where surface motion is not well-defined. <br/>Image credit <a href='https://unsplash.com/photos/LfvXuPjNGhE/'>Claire Rowlett</a>.",
      "animation/image_26-0-trajectories.mp4",
      "animation/image_26-0-warped.mp4",
      "animation/image_26-0-holefill.mp4",
      "animation/image_26-1-trajectories.mp4",
      "animation/image_26-1-warped.mp4",
      "animation/image_26-1-holefill.mp4",
    ],
    [
      "animation/image_65.png",
      "In this example, the model reproduces some plausible digging motions, segmenting the hands from the background and from each other.  In the top example, the two hands move independently and one occludes the other, whereas in the bottom, they move together.  Note that the plants also occlude the hands in the bottom example, suggesting an understanding of the scene depth and the underlying activity. <br/>Image credit <a href='https://unsplash.com/photos/4z3lnwEvZQw/'>Jonathan Kemper</a>.",
      "animation/image_65-0-trajectories.mp4",
      "animation/image_65-0-warped.mp4",
      "animation/image_65-0-holefill.mp4",
      "animation/image_65-1-trajectories.mp4",
      "animation/image_65-1-warped.mp4",
      "animation/image_65-1-holefill.mp4",
    ],

    [
      "animation/image_135.png",
      "For this dog, the top example shows a biting and dragging motion where the dog largely remains still (note the slight compression of the ball).  For the bottom example, the dog moves its right paw and scoops toward the ball with its mouth.  Although the motions are not as fluid as some of the others, the segmentation is plausible for a pair of interacting objects. <br/>Image credit <a href='https://unsplash.com/photos/J40C1k6Fut0/'>Tatiana Rodriguez</a>.",
      "animation/image_135-0-trajectories.mp4",
      "animation/image_135-0-warped.mp4",
      "animation/image_135-0-holefill.mp4",
      "animation/image_135-1-trajectories.mp4",
      "animation/image_135-1-warped.mp4",
      "animation/image_135-1-holefill.mp4",
    ],
    [
      "animation/normal-cat.png",
      "We see that the cat is segmented well from the background in both cases, and the tail moves independently of the rest of the cat.  In the top example, the cat looks to the right and crouches slightly, whereas in on the bottom, it moves its left leg.  In general, the model still struggles to make animals take steps, probably because they happen quickly in real videos and aren't well-represented in our training set, but the model does seem to be aware of their limbs. <br/>Image credit <a href='https://unsplash.com/photos/p6yH8VmGqxo/'>Kabo</a>.",
      "animation/normal-cat-0-trajectories.mp4",
      "animation/normal-cat-0-warped.mp4",
      "animation/normal-cat-0-holefill.mp4",
      "animation/normal-cat-5-trajectories.mp4",
      "animation/normal-cat-5-warped.mp4",
      "animation/normal-cat-5-holefill.mp4",
    ],
    [
      "animation/brown-bird.png",
      "Birds are somewhat un in the training set, but the model can nonetheless produce some plausible motion.  In the top example, the wings flex, and in the bottom example, the bird glides relative to the background. <br/>Image credit <a href='https://unsplash.com/photos/hjRVSBnA25w/'>Emanuel Antonov</a>.",
      "animation/brown-bird-6-trajectories.mp4",
      "animation/brown-bird-6-warped.mp4",
      "animation/brown-bird-6-holefill.mp4",
      "animation/brown-bird-19-trajectories.mp4",
      "animation/brown-bird-19-warped.mp4",
      "animation/brown-bird-19-holefill.mp4",
    ],
    [
      "animation/image_8.png",
      "In this example, the model demonstrates some understanding of the activity and the way hands are used.  The top example shows the beginning of a slice with the knife as the hand lifts out of the way, while the bottom shows the hand gathering the vegetables with only a slight motion of the hand and knife. Note that even in the bottom example, the slight hand motion roughly matches the slight knife motion. <br/>Image credit <a href='https://unsplash.com/photos/yWG-ndhxvqY/'>Alyson McPhee</a>.",
      "animation/image_8-0-trajectories.mp4",
      "animation/image_8-0-warped.mp4",
      "animation/image_8-0-holefill.mp4",
      "animation/image_8-1-trajectories.mp4",
      "animation/image_8-1-warped.mp4",
      "animation/image_8-1-holefill.mp4",
    ],
    [
      "animation/image_18.png",
      "This photo was likely taken from a drone or other aircraft, and the model imitates the motion that would be seen in such arial footage: in both cases, the camera moves toward the scene, but at slightly different angles.  Note that the model does segment the building from the background; in the patch-based-warping view for both examples, we can see a slight occlusion boundary appear at the top of the building, but not the bottom, which would be correct for a moving camera. <br/>Image credit <a href='https://unsplash.com/photos/MGCQ_5bgJXY/'>Athina Vrikki</a>.",
      "animation/image_18-0-trajectories.mp4",
      "animation/image_18-0-warped.mp4",
      "animation/image_18-0-holefill.mp4",
      "animation/image_18-1-trajectories.mp4",
      "animation/image_18-1-warped.mp4",
      "animation/image_18-1-holefill.mp4",
    ],
    [
      "animation/image_25.png",
      "In the top row, both of the people slightly rock on their feet and move closer together, while in the bottom row, both of them slightly lean rightward.  Note that although the bottom video's pixel reconstruction is slightly messy, the underlying trajectories are plausible for the arms. <br/>Image credit <a href='https://unsplash.com/photos/swi1DGRCshQ/'>Christina &#64; wocintechchat.com</a>.",
      "animation/image_25-0-trajectories.mp4",
      "animation/image_25-0-warped.mp4",
      "animation/image_25-0-holefill.mp4",
      "animation/image_25-1-trajectories.mp4",
      "animation/image_25-1-warped.mp4",
      "animation/image_25-1-holefill.mp4",
    ],
    [
      "animation/image_63.png",
      "The camera moves slowly forward and towards right in the top example as the dog points its nose toward the camera. In the bottom example, the dog cocks its head.  On the bottom, the collar produces a spurious occlusion that might be filled in with fur given a better hole-filling model, but in this case it's instead filled in temporarily with background. <br/>Image credit <a href='https://unsplash.com/photos/EUnFBfDgJjY/'>Jen Vazquez</a>.",
      "animation/image_63-0-trajectories.mp4",
      "animation/image_63-0-warped.mp4",
      "animation/image_63-0-holefill.mp4",
      "animation/image_63-1-trajectories.mp4",
      "animation/image_63-1-warped.mp4",
      "animation/image_63-1-holefill.mp4",
    ],
    [
      "animation/portrait.png",
      "Our diffusion models are not trained on paintings; animations of paintings like this are rare in real-world videos.  Nevertheless, the model can recognize the presence of a person and produce plausible animations.  In the top example, the subject moves as if he is straightening up. In the bottom example, the subject is bowing his head. <br/>Image credit <a href='https://unsplash.com/photos/Krm-9syUmOY/'>Europeana</a>.",
      "animation/portrait-15-trajectories.mp4",
      "animation/portrait-15-warped.mp4",
      "animation/portrait-15-holefill.mp4",
      "animation/portrait-34-trajectories.mp4",
      "animation/portrait-34-warped.mp4",
      "animation/portrait-34-holefill.mp4",
    ],
  ]
  document.getElementById("input").src=("https://storage.googleapis.com/dm-tapnet/tapir-blogpost/videos/"+data[idx][0]);
  document.getElementById("text").innerHTML=data[idx][1];
  for (var i = 0; i<6; ++i){
    vid=document.getElementById("vid"+i);
    vid.getElementsByTagName('source')[0].src=("https://storage.googleapis.com/dm-tapnet/tapir-blogpost/videos/"+data[idx][i+2]);
    //vid.width=224
    //vid.height=224
    vid.load();
  }
}

function set_source2(idx){
  data=[
    [
        "The task is to <b>glue the 2 blocks together</b>. The robot has to apply glue to a wooden block, place another wooden block on top, push them together, and place them to the right of the white gear. Masking tape, tapir, balls, apple, and sticky note were not present in the demos.",
        "gluing_11.mp4",
        "gluing_15.mp4",
        "gluing_17.mp4",
    ],
    [
        "The task is to <b>pick up the the apple and place it on the jello</b>. Only the apple and jello were present in the demos.  RoboTAP works even if there are other red objects on the scene, the goal is partially occluded or when the scene is full of distracting objects.",
        "apple_on_jello_1.mp4",
        "apple_on_jello_10.mp4",
        "apple_on_jello_14.mp4",
    ],
    [
        "The task is to <b>pick up the green juggling ball and place it on the blue one</b>. RoboTAP succeeds, despite both of these juggling balls being mostly textureless, deformable and symmetric. We can also place the objects on top of of jello or tomato without the need to modify RoboTAP.  Only the juggling balls were present in the demos.",
        "juggling_stack_20.mp4",
        "juggling_stack_25.mp4",
        "juggling_stack_29.mp4",
    ],
    [
        "The task is to <b>insert the 4 wooden objects</b> into their sockets. This is a long horizon task which involves textureless and symmetric objects and requires precise placement. RoboTAP can generalize to novel starting positions.  Only the stencil and the four objects were present in the demos.",
        "four_block_stencil_w2_39.mp4",
        "four_block_stencil_w2_47.mp4",
        "four_block_stencil_w2_49.mp4",
    ],
    [
        "The task is to <b>attach the orange brick across the blue brick</b>. RoboTAP can correctly identify the objects even when partially occluded.  Only legos were present in the demos.",
        "lego_stack_w3_2.mp4",
        "lego_stack_w3_11.mp4",
        "lego_stack_w3_12.mp4",
    ],
    [
        "The task is to <b>pick up the tapir toy and bring it over to the side of the wooden robot</b>. Only the tapir and the robot were present in the demos.  Thanks to TAPIR's ability to track points RoboTAP can correctly identify the robot even when other objects with a very similar texture are in the scene.",
        "tapir_robot_v2_4.mp4",
        "tapir_robot_v2_14.mp4",
        "tapir_robot_v2_16.mp4",
    ],
    [
        "The task is to <b>stack 4 wooden objects</b> on top of each other. This task is particularly challanging, because any errors in placement compound and can make the final tower fall over.  Clutter was not evaluated thoroughly, as we found that occasional false-positives from TAPIR slightly reduced the consistency of grasps and placements, which tended to accumulate and resulted in unsuccessful stacks.",
        "four_block_stack_1.mp4",
        "four_block_stack_3.mp4",
        "four_block_stack_8.mp4",
    ],
    [
        "The task is to <b>pick up the butter and place it in the hand</b>. Since RoboTAP works by aligning arbitrary points, we created a task where the location of the hand is important. We did not use any specific hand tracking solution.  Only the hand and the butter were present in the demos.",
        "pass_butter_1.mp4",
        "pass_butter_20.mp4",
        "pass_butter_22.mp4",
    ],
    [
        "The task is to <b>pick up the gear and place it on the grid</b>. This task was primarily used to evaluate the precision of the placement. In these videos we can see the 3 evaluation settings further described in the paper.",
        "precision_1.mp4",
        "precision_11.mp4",
        "precision_12.mp4",
    ],

  ]
  document.getElementById("success_text").innerHTML=data[idx][0];
  for (var i = 0; i<3; ++i){
    vid=document.getElementById("success_vid"+i);
    vid.getElementsByTagName('source')[0].src=("https://storage.googleapis.com/dm-tapnet/robotap/videos/success_gallery/"+data[idx][i+1]);
    vid.load();
  }
}

