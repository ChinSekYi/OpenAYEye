import { Box } from "@mui/material";
import { Header, HeatMap } from "../../components";

const Heat = () => {
  return (
    <Box m="20px">
      <Header title="Heat Map" subtitle="Simple Heat Map" />
      <Box height="75vh">
        <HeatMap />
      </Box>
    </Box>
  );
};

export default Heat;
