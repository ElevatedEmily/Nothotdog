import React, { useState } from "react";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import axios from "axios";
import Webcam from "react-webcam";
import { Box, Button, Typography, Grid, AppBar, Toolbar } from "@mui/material";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import CancelIcon from "@mui/icons-material/Cancel";

function Home() {
  const [file, setFile] = useState(null);
  const [imageSrc, setImageSrc] = useState(null);
  const [prediction, setPrediction] = useState("");
  const [isCamera, setIsCamera] = useState(false);

  const webcamRef = React.useRef(null);

  const handleFileChange = (e) => {
    const uploadedFile = e.target.files[0];
    setFile(uploadedFile);
    const reader = new FileReader();
    reader.onloadend = () => setImageSrc(reader.result);
    reader.readAsDataURL(uploadedFile);
  };

  const captureImage = () => {
    const image = webcamRef.current.getScreenshot();
    setImageSrc(image);
    setFile(dataURLtoFile(image, "captured_image.jpeg"));
  };

  const dataURLtoFile = (dataUrl, filename) => {
    const arr = dataUrl.split(",");
    const mime = arr[0].match(/:(.*?);/)[1];
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
    while (n--) {
      u8arr[n] = bstr.charCodeAt(n);
    }
    return new File([u8arr], filename, { type: mime });
  };

  const handleSubmit = async () => {
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post("https://hotdog-nothotdog-b2ca2.cloudfunctions.net/api/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setPrediction(response.data.prediction);
    } catch (error) {
      console.error("Error during prediction:", error);
      setPrediction("Error during prediction");
    }
  };

  const renderBanner = () => {
    if (!prediction) return null;

    const isHotdog = prediction === "Negative"; // Assuming Negative means "Hotdog"

    return (
      <Box
        sx={{
          position: "absolute",
          top: 0,
          left: 0,
          width: "100%",
          height: "15%", // Covers the top fifth of the image
          backgroundColor: isHotdog ? "green" : "red",
          color: "white",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          zIndex: 2,
        }}
      >
        <Typography
          variant="h5"
          sx={{ fontWeight: "bold", fontSize: "2rem", marginBottom: "5px" }}
        >
          {isHotdog ? "Hotdog" : "Not Hotdog"}
        </Typography>
        <Box
          sx={{
            position: "relative",
            backgroundColor: "white",
            color: isHotdog ? "green" : "red",
            borderRadius: "50%",
            width: "35px", // Thinner circle
            height: "35px",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            marginTop: "-10px", // Moved further down
            zIndex: 3,
            boxShadow: "0px 4px 6px rgba(0, 0, 0, 0.2)",
          }}
        >
          {isHotdog ? (
            <CheckCircleIcon sx={{ fontSize: "35px" }} />
          ) : (
            <CancelIcon sx={{ fontSize: "35px" }} />
          )}
        </Box>
      </Box>
    );
  };

  return (
    <Box
      sx={{
        width: "60vw",
        height: "auto",
        margin: "auto",
        position: "absolute",
        top: "50%",
        left: "50%",
        transform: "translate(-50%, -50%)",
        padding: "20px",
        backgroundColor: "#f9f9f9",
        borderRadius: "15px",
        boxShadow: "0 4px 10px rgba(0, 0, 0, 0.1)",
        textAlign: "center",
      }}
    >
      <Grid container spacing={2} justifyContent="center">
        <Grid item xs={12}>
          <Box
            sx={{
              position: "relative", // Necessary for the banner overlay
              backgroundColor: "#e0e0e0",
              borderRadius: "15px",
              overflow: "hidden",
              width: "100%",
              maxHeight: "70vh",
            }}
          >
            {imageSrc ? (
              <img
                src={imageSrc}
                alt="Uploaded"
                style={{
                  width: "100%",
                  height: "auto",
                }}
              />
            ) : (
              <Typography
                variant="body1"
                sx={{
                  textAlign: "center",
                  color: "#888",
                  padding: "50px",
                }}
              >
                No image selected
              </Typography>
            )}
            {renderBanner()}
          </Box>
        </Grid>
      </Grid>

      <Box sx={{ marginTop: "20px" }}>
        <Button
          variant="outlined"
          sx={{ color: "#F984AD", marginRight: "10px" }}
          onClick={() => setIsCamera(!isCamera)}
        >
          {isCamera ? "Stop Camera" : "Use Camera"}
        </Button>
        {!isCamera && (
          <Button
            variant="contained"
            component="label"
            sx={{ backgroundColor: "#F984AD", marginRight: "10px" }}
          >
            Upload Image
            <input type="file" hidden accept="image/*" onChange={handleFileChange} />
          </Button>
        )}
        <Button
          variant="contained"
          sx={{ backgroundColor: "#F984AD", marginRight: "10px" }}
          disabled={!file}
          onClick={handleSubmit}
        >
          Predict
        </Button>
      </Box>
    </Box>
  );
}

function About() {
  return (
    <Box sx={{ padding: "20px", textAlign: "center" }}>
      <Typography variant="h4" gutterBottom>
        About the Project
      </Typography>
      <Typography variant="body1" paragraph>
        This is a binary classifier built using the ResNet-50 model trained on image 
        of hotdogs and not hotdogs. The website it built using Flask and React. 
      </Typography>
      <Typography variant="h4" gutterBottom>
        So, it only does hot dogs?
      </Typography>
      <Typography variant="body1" paragraph>
        No, it also does not hot dog!
      </Typography>
      <Typography variant="body1">
        Created by <strong>Emily Andrews</strong>. Explore more projects at my{" "}
        <a
          href="https://github.com/ElevatedEmily"
          target="_blank"
          rel="noopener noreferrer"
          style={{ color: "blue", textDecoration: "underline" }}
        >
          GitHub
        </a> 
        .
      </Typography>
    </Box>
  );
}

function App() {
  return (
    <Router>
      <AppBar position="static" sx={{ marginBottom: "20px",  backgroundColor: "#F984AD" }}>
        <Toolbar>
          <Typography variant="h6" sx={{ flexGrow: 1, color: "fff" }}>
            Hotdog Not Hotdog
          </Typography>
          <Button color="#F984AD" component={Link} to="/">
            Home
          </Button>
          <Button color="#F984AD" component={Link} to="/about">
            About
          </Button>
        </Toolbar>
      </AppBar>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/about" element={<About />} />
      </Routes>
    </Router>
  );
}

export default App;
